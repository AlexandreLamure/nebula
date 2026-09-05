#ifndef SCENE_H
#define SCENE_H

#include <SceneObject.h>
#include <PointLight.h>
#include <Camera.h>
#include <TypedBuffer.h>
#include <shaderStructs.h>

#include <vector>
#include <memory>

namespace nebula {

class Scene : NonMovable {

    public:
        Scene();

        static Result<std::unique_ptr<Scene>> fromGltf(const std::string& fileName);

        void prepareFrame();
        void renderDepth() const;
        void render();

        void addObject(SceneObject obj);
        void addLight(PointLight obj);

        Span<const SceneObject> objects() const;
        Span<const PointLight> pointLights() const;

        Camera& camera();
        const Camera& camera() const;

        void setEnvmap(std::shared_ptr<Texture> env);
        void setIblIntensity(float intensity);

        void setSun(float altitude, float azimuth, glm::vec3 color = glm::vec3(1.0f));

    private:
        void prepareObjects();

        std::vector<SceneObject> _objects;
        std::vector<PointLight> _pointLights;

        std::vector<u32> _opaqueDraws;
        std::vector<u32> _transparentDraws;

        glm::vec3 _sunDirection = glm::vec3(0.2f, 1.0f, 0.1f);
        glm::vec3 _sunColor = glm::vec3(1.0f);

        std::shared_ptr<Texture> _envmap;
        float _iblIntensity = 1.0f;
        Material _skyMaterial;
        std::shared_ptr<Program> _depthProgram;
        std::shared_ptr<Program> _depthAlphaTestProgram;

        TypedBuffer<shader::FrameData> _frameUbo;
        TypedBuffer<shader::PointLight> _lightBuffer;

        Camera _camera;
};

}

#endif // SCENE_H
