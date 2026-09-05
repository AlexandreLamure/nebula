#include "Scene.h"

#include "VkContext.h"
#include "ThreadPool.h"

#include <algorithm>

namespace nebula {

Scene::Scene() {
    _skyMaterial.setProgram(Program::fromFiles("screen.slang", "sky.slang"));
    _skyMaterial.setDepthTestMode(DepthTestMode::None);

    _depthProgram = Program::fromFiles("basic.slang", "depth.slang");
    _depthAlphaTestProgram = Program::fromFiles("basic.slang", "depth_ALPHA_TEST.slang");

    _envmap = std::make_shared<Texture>(Texture::emptyCubemap(4, ImageFormat::RGBA8_UNORM));
}

void Scene::addObject(SceneObject obj) {
    _objects.emplace_back(std::move(obj));
}

void Scene::addLight(PointLight obj) {
    _pointLights.emplace_back(std::move(obj));
}

Span<const SceneObject> Scene::objects() const {
    return _objects;
}

Span<const PointLight> Scene::pointLights() const {
    return _pointLights;
}

Camera& Scene::camera() {
    return _camera;
}

const Camera& Scene::camera() const {
    return _camera;
}

void Scene::setEnvmap(std::shared_ptr<Texture> env) {
    _envmap = std::move(env);
}

void Scene::setIblIntensity(float intensity) {
    _iblIntensity = intensity;
}

void Scene::setSun(float altitude, float azimuth, glm::vec3 color) {
    // Convert from degrees to radians
    const float alt = glm::radians(altitude);
    const float azi = glm::radians(azimuth);
    // Convert from polar to cartesian
    _sunDirection = glm::vec3(sin(azi) * cos(alt), sin(alt), cos(azi) * cos(alt));
    _sunColor = color;
}

void Scene::prepareObjects() {
    const u32 count = u32(_objects.size());
    const Frustum frustum = _camera.buildFrustum();

    std::vector<Sphere> boundsWs(count);
    std::vector<u8> visible(count);

    parallelFor(count, [&](u32 i) {
        boundsWs[i] = _objects[i].computeBoundingSphereWs();
        visible[i] = frustumSphereIntersection(frustum, boundsWs[i]) ? 1u : 0u;
    });

    _opaqueDraws.clear();
    _transparentDraws.clear();
    _opaqueDraws.reserve(count);
    _transparentDraws.reserve(count);

    for(u32 i = 0; i != count; ++i) {
        if(!visible[i]) {
            continue;
        }
        if(_objects[i].material().isOpaque()) {
            _opaqueDraws.push_back(i);
        } else {
            _transparentDraws.push_back(i);
        }
    }

    // Group by pipeline, then material (descriptors), then mesh (VBO/IBO).
    std::sort(_opaqueDraws.begin(), _opaqueDraws.end(), [this](u32 a, u32 b) {
        const SceneObject& oa = _objects[a];
        const SceneObject& ob = _objects[b];
        const Program* pa = &oa.material().program();
        const Program* pb = &ob.material().program();
        if(pa != pb) {
            return pa < pb;
        }
        const Material* ma = &oa.material();
        const Material* mb = &ob.material();
        if(ma != mb) {
            return ma < mb;
        }
        return &oa.mesh() < &ob.mesh();
    });

    // Back-to-front so alpha blending composites correctly.
    const glm::vec3 camPos = _camera.position();
    std::sort(_transparentDraws.begin(), _transparentDraws.end(), [&](u32 a, u32 b) {
        const glm::vec3 da = boundsWs[a]._center - camPos;
        const glm::vec3 db = boundsWs[b]._center - camPos;
        const float distA = glm::dot(da, da);
        const float distB = glm::dot(db, db);
        if(distA != distB) {
            return distA > distB;
        }
        return a < b;
    });
}

void Scene::prepareFrame() {
    // Recreated each frame; ~ByteBuffer defers GPU free until this frame's fence
    _frameUbo = TypedBuffer<shader::FrameData>(nullptr, 1);
    {
        auto mapping = _frameUbo.map(AccessType::WriteOnly);
        mapping[0].camera.viewProj = _camera.viewProjMatrix();
        mapping[0].camera.invViewProj = glm::inverse(_camera.viewProjMatrix());
        mapping[0].camera.position = _camera.position();
        mapping[0].pointLightCount = u32(_pointLights.size());
        mapping[0].sunColor = _sunColor;
        mapping[0].sunDir = glm::normalize(_sunDirection);
        mapping[0].iblIntensity = _iblIntensity;
    }

    _lightBuffer = TypedBuffer<shader::PointLight>(nullptr, std::max(_pointLights.size(), size_t(1)));
    {
        auto mapping = _lightBuffer.map(AccessType::WriteOnly);
        for(size_t i = 0; i != _pointLights.size(); ++i) {
            const auto& light = _pointLights[i];
            mapping[i] = {
                light.position(),
                light.radius(),
                light.color(),
                0.0f
            };
        }
    }

    DEBUG_ASSERT(_envmap && !_envmap->isNull());
    bindFrame({
        .ubo = _frameUbo.vkBuffer(),
        .uboSize = _frameUbo.byteSize(),
        .lights = _lightBuffer.vkBuffer(),
        .lightsSize = _lightBuffer.byteSize(),
        .env = _envmap.get(),
        .brdf = &brdfLut(),
    });

    prepareObjects();
}

void Scene::renderDepth() const {
    DEBUG_ASSERT(_frameUbo.vkBuffer() && _depthProgram && _depthAlphaTestProgram);

    for(const u32 i : _opaqueDraws) {
        const SceneObject& obj = _objects[i];

        RasterState raster = obj.material().rasterState();
        raster.alphaBlend = false;
        raster.depthTestEnable = true;
        raster.depthWriteEnable = true;
        raster.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

        // Override the shader to only write depth
        const Program& program = obj.material().isAlphaTested()
            ? *_depthAlphaTestProgram
            : *_depthProgram;
        obj.render(program, raster);
    }
}

void Scene::render() {
    DEBUG_ASSERT(_frameUbo.vkBuffer());

    for(const u32 i : _opaqueDraws) {
        const SceneObject& obj = _objects[i];
        // Override the depth state. Depth is already written by the z-prepass.
        RasterState raster = obj.material().rasterState();
        raster.depthTestEnable = true;
        raster.depthWriteEnable = false;
        raster.depthCompareOp = VK_COMPARE_OP_EQUAL;
        obj.render(obj.material().program(), raster);
    }

    // Draw sky
    {
        PushConstants push = _skyMaterial.buildPushConstants();
        push.set(HASH("intensity"), _iblIntensity);
        // After the z-prepass, leftover pixels still have the reverse-Z far value (0).
        // The fullscreen sky triangle sits at z=0, so Equal shades only uncovered pixels.
        RasterState raster = _skyMaterial.rasterState();
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.depthTestEnable = true;
        raster.depthWriteEnable = false;
        raster.depthCompareOp = VK_COMPARE_OP_EQUAL;
        drawFullscreen(_skyMaterial.program(), raster, _skyMaterial.passResources(), push);
    }

    for(const u32 i : _transparentDraws) {
        _objects[i].render();
    }

    // Release this frame's UBOs so ~ByteBuffer queues them on this frame's fence.
    _frameUbo = {};
    _lightBuffer = {};
}

}
