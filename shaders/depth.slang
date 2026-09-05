#ifdef ALPHA_TEST
import structs;

[[vk::binding(0, 1)]] Sampler2D inTexture;

[shader("fragment")]
void fragmentMain([[vk::location(1)]] float2 uv : UV) {
    if(inTexture.Sample(uv).a <= pc.alphaCutoff) {
        discard;
    }
}
#else
[shader("fragment")]
void fragmentMain() {
}
#endif
