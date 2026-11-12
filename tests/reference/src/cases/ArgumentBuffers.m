// Argument buffer cases: encoders made from a function and from argument
// descriptors, buffers, textures, samplers and constants encoded and read
// by a kernel, arrays of resources, nested argument buffers (tier 2) and
// argument buffers written directly with GPU addresses and resource IDs
// (Metal 3, tier 2).

#import "WMRefCase.h"
#import "LibrariesSupport.h"

typedef struct {
    id<MTLBuffer> values;
    id<MTLTexture> image;
    id<MTLSamplerState> sampler;
    id<MTLBuffer> out;
} WMArgResources;

// 16 small integer values in a buffer and in a 4x4 R32Float texture;
// products of small integers are exact in binary32.
static WMArgResources WMMakeArgResources(WMCase *c) {
    WMArgResources r;
    float values[16], texels[16];
    for (int i = 0; i < 16; i++) {
        values[i] = (float)(i + 1);
        texels[i] = (float)(100 + i * 3);
    }
    r.values = [c bufferWithBytes:values length:sizeof values];
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                          width:4 height:4
                                                                                      mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderRead;
    r.image = [c texture:descriptor];
    [r.image replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:texels bytesPerRow:16];
    MTLSamplerDescriptor *sampler = [MTLSamplerDescriptor new];
    sampler.minFilter = MTLSamplerMinMagFilterNearest;
    sampler.magFilter = MTLSamplerMinMagFilterNearest;
    sampler.supportArgumentBuffers = YES;
    r.sampler = [c sampler:sampler];
    r.out = [c buffer:16 * 3 * sizeof(uint32_t)];
    return r;
}

static void WMEncodeArgResources(id<MTLArgumentEncoder> encoder, id<MTLBuffer> argumentBuffer, WMArgResources r) {
    [encoder setArgumentBuffer:argumentBuffer offset:0];
    [encoder setBuffer:r.values offset:0 atIndex:0];
    [encoder setTexture:r.image atIndex:1];
    [encoder setSamplerState:r.sampler atIndex:2];
    *(uint32_t *)[encoder constantDataAtIndex:3] = 3;
    [encoder setBuffer:r.out offset:0 atIndex:4];
}

static void WMRunArgRead(WMCase *c, id<MTLComputePipelineState> pipeline, id<MTLBuffer> argumentBuffer,
                         WMArgResources r) {
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setBuffer:argumentBuffer offset:0 atIndex:0];
        [e useResource:r.values usage:MTLResourceUsageRead];
        [e useResource:r.image usage:MTLResourceUsageRead];
        [e useResource:r.out usage:MTLResourceUsageWrite];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
    }];
}

WMREF_CASE(argumentBufferFromFunction, "argument-buffer.encoder-from-function", "argument-buffer.encoding",
           "newArgumentEncoderWithBufferIndex: on a kernel; buffer, texture, sampler and constant read through it")
{
    id<MTLFunction> function = [c function:@"arg_read"];
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"arg_read"];
    id<MTLArgumentEncoder> encoder = [function newArgumentEncoderWithBufferIndex:0];
    if (!encoder) [c fail:@"newArgumentEncoderWithBufferIndex returned nil"];
    [c value:@"encoded-length" u64:encoder.encodedLength rule:WMDevice()];
    [c value:@"alignment" u64:encoder.alignment rule:WMDevice()];
    WMArgResources r = WMMakeArgResources(c);
    id<MTLBuffer> argumentBuffer = [c buffer:encoder.encodedLength];
    WMEncodeArgResources(encoder, argumentBuffer, r);
    WMRunArgRead(c, pipeline, argumentBuffer, r);
    [c data:@"results" elem:WMElemU32 bytes:r.out.contents length:r.out.length rule:WMExact()];
}

WMREF_CASE(argumentBufferFromDescriptors, "argument-buffer.encoder-from-descriptors", "argument-buffer.encoding",
           "newArgumentEncoderWithArguments: with argument descriptors matching a kernel's argument buffer")
{
    NSMutableArray<MTLArgumentDescriptor *> *arguments = [NSMutableArray array];
    MTLArgumentDescriptor *values = [MTLArgumentDescriptor argumentDescriptor];
    values.dataType = MTLDataTypePointer;
    values.index = 0;
    values.access = MTLBindingAccessReadOnly;
    [arguments addObject:values];
    MTLArgumentDescriptor *image = [MTLArgumentDescriptor argumentDescriptor];
    image.dataType = MTLDataTypeTexture;
    image.index = 1;
    image.textureType = MTLTextureType2D;
    image.access = MTLBindingAccessReadOnly;
    [arguments addObject:image];
    MTLArgumentDescriptor *sampler = [MTLArgumentDescriptor argumentDescriptor];
    sampler.dataType = MTLDataTypeSampler;
    sampler.index = 2;
    [arguments addObject:sampler];
    MTLArgumentDescriptor *scale = [MTLArgumentDescriptor argumentDescriptor];
    scale.dataType = MTLDataTypeUInt;
    scale.index = 3;
    [arguments addObject:scale];
    MTLArgumentDescriptor *out = [MTLArgumentDescriptor argumentDescriptor];
    out.dataType = MTLDataTypePointer;
    out.index = 4;
    out.access = MTLBindingAccessReadWrite;
    [arguments addObject:out];

    id<MTLArgumentEncoder> encoder = [c.device newArgumentEncoderWithArguments:arguments];
    if (!encoder) [c fail:@"newArgumentEncoderWithArguments returned nil"];
    id<MTLArgumentEncoder> fromFunction = [[c function:@"arg_read"] newArgumentEncoderWithBufferIndex:0];
    [c value:@"encoded-length" u64:encoder.encodedLength rule:WMDevice()];
    [c value:@"alignment" u64:encoder.alignment rule:WMDevice()];
    [c value:@"same-length-as-function-encoder" boolean:encoder.encodedLength == fromFunction.encodedLength
        rule:WMDevice()];
    WMArgResources r = WMMakeArgResources(c);
    id<MTLBuffer> argumentBuffer = [c buffer:encoder.encodedLength];
    WMEncodeArgResources(encoder, argumentBuffer, r);
    WMRunArgRead(c, [c computePipeline:@"arg_read"], argumentBuffer, r);
    [c data:@"results" elem:WMElemU32 bytes:r.out.contents length:r.out.length rule:WMExact()];
}

WMREF_CASE(argumentBufferArrays, "argument-buffer.arrays", "argument-buffer.encoding",
           "Arrays of textures, buffers and constants in an argument buffer, set with range setters")
{
    id<MTLFunction> function = [c function:@"arg_arrays"];
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"arg_arrays"];
    id<MTLArgumentEncoder> encoder = [function newArgumentEncoderWithBufferIndex:0];
    if (!encoder) [c fail:@"newArgumentEncoderWithBufferIndex returned nil"];
    [c value:@"encoded-length" u64:encoder.encodedLength rule:WMDevice()];

    id<MTLTexture> textures[4];
    for (int t = 0; t < 4; t++) {
        MTLTextureDescriptor *descriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float width:2 height:1
                                                           mipmapped:NO];
        descriptor.usage = MTLTextureUsageShaderRead;
        textures[t] = [c texture:descriptor];
        const float texels[2] = {(float)(10 * t), (float)(10 * t + 1)};
        [textures[t] replaceRegion:MTLRegionMake2D(0, 0, 2, 1) mipmapLevel:0 withBytes:texels bytesPerRow:8];
    }
    id<MTLBuffer> buffers[3];
    NSUInteger offsets[3] = {0, 0, 0};
    for (int b = 0; b < 3; b++) {
        uint32_t words[16];
        for (int i = 0; i < 16; i++) words[i] = (uint32_t)(1000 * (b + 1) + i);
        buffers[b] = [c bufferWithBytes:words length:sizeof words];
    }
    id<MTLBuffer> argumentBuffer = [c buffer:encoder.encodedLength];
    [encoder setArgumentBuffer:argumentBuffer offset:0];
    [encoder setTextures:textures withRange:NSMakeRange(0, 4)];
    [encoder setBuffers:buffers offsets:offsets withRange:NSMakeRange(4, 3)];
    *(uint32_t *)[encoder constantDataAtIndex:7] = 0xaaaa;
    *(uint32_t *)[encoder constantDataAtIndex:8] = 0xbbbb;
    id<MTLBuffer> out = [c buffer:12 * 3 * sizeof(uint32_t)];
    id<MTLTexture> t0 = textures[0], t1 = textures[1], t2 = textures[2], t3 = textures[3];
    id<MTLBuffer> b0 = buffers[0], b1 = buffers[1], b2 = buffers[2];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setBuffer:argumentBuffer offset:0 atIndex:0];
        [e setBuffer:out offset:0 atIndex:1];
        for (id<MTLResource> resource in @[t0, t1, t2, t3, b0, b1, b2])
            [e useResource:resource usage:MTLResourceUsageRead];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(12, 1, 1)];
    }];
    [c data:@"results" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];
}

WMREF_CASE(argumentBufferNested, "argument-buffer.nested", "argument-buffer.tier2",
           "An argument buffer holding a pointer to another argument buffer (argument buffers tier 2)")
{
    [c require:@"argument-buffers-tier2" value:c.device.argumentBuffersSupport == MTLArgumentBuffersTier2];
    id<MTLFunction> function = [c function:@"arg_nested"];
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"arg_nested"];
    id<MTLArgumentEncoder> outer = [function newArgumentEncoderWithBufferIndex:0];
    id<MTLArgumentEncoder> inner = [outer newArgumentEncoderForBufferAtIndex:0];
    if (!outer || !inner) [c fail:@"argument encoders returned nil"];
    [c value:@"outer.encoded-length" u64:outer.encodedLength rule:WMDevice()];
    [c value:@"inner.encoded-length" u64:inner.encodedLength rule:WMDevice()];

    uint32_t words[8];
    for (int i = 0; i < 8; i++) words[i] = (uint32_t)(i * i);
    id<MTLBuffer> data = [c bufferWithBytes:words length:sizeof words];
    id<MTLBuffer> innerBuffer = [c buffer:inner.encodedLength];
    [inner setArgumentBuffer:innerBuffer offset:0];
    [inner setBuffer:data offset:0 atIndex:0];
    id<MTLBuffer> outerBuffer = [c buffer:outer.encodedLength];
    [outer setArgumentBuffer:outerBuffer offset:0];
    [outer setBuffer:innerBuffer offset:0 atIndex:0];
    *(uint32_t *)[outer constantDataAtIndex:1] = 500;
    id<MTLBuffer> out = [c buffer:sizeof words];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setBuffer:outerBuffer offset:0 atIndex:0];
        [e setBuffer:out offset:0 atIndex:1];
        [e useResource:innerBuffer usage:MTLResourceUsageRead];
        [e useResource:data usage:MTLResourceUsageRead];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(8, 1, 1)];
    }];
    [c data:@"results" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];
}

// With Metal 3 and argument buffers tier 2, an argument buffer is a plain
// structure: a device pointer is the buffer's gpuAddress (64 bits) and a
// texture is its gpuResourceID (MTLResourceID, 64 bits).
static NSString *const WMDirectArgumentSource =
    @"#include <metal_stdlib>\nusing namespace metal;\n"
     "struct Direct {\n"
     "    device const uint *data;\n"
     "    texture2d<float> image;\n"
     "    uint bias;\n"
     "};\n"
     "kernel void src_direct(constant Direct &args [[buffer(0)]], device uint *out [[buffer(1)]],\n"
     "                       uint i [[thread_position_in_grid]])\n"
     "{\n"
     "    out[i * 2u] = args.data[i] + args.bias;\n"
     "    out[i * 2u + 1u] = as_type<uint>(args.image.read(uint2(i, 0u)).x);\n"
     "}\n";

WMREF_CASE(argumentBufferDirect, "argument-buffer.gpu-address", "argument-buffer.tier2",
           "An argument buffer written by the CPU with gpuAddress and gpuResourceID (Metal 3, tier 2)")
{
    [c require:@"family-metal3" value:[c.device supportsFamily:MTLGPUFamilyMetal3]];
    [c require:@"argument-buffers-tier2" value:c.device.argumentBuffersSupport == MTLArgumentBuffersTier2];
    if (@available(macOS 13.0, iOS 16.0, *)) {
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_0;
        NSError *error = nil;
        id<MTLLibrary> library = [c.device newLibraryWithSource:WMDirectArgumentSource options:options error:&error];
        if (!library) [c fail:@"direct argument library: %@", error];
        id<MTLComputePipelineState> pipeline =
            [c.device newComputePipelineStateWithFunction:[library newFunctionWithName:@"src_direct"] error:&error];
        if (!pipeline) [c fail:@"direct argument pipeline: %@", error];

        uint32_t words[4] = {7, 70, 700, 7000};
        id<MTLBuffer> data = [c bufferWithBytes:words length:sizeof words];
        MTLTextureDescriptor *descriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float width:4 height:1
                                                           mipmapped:NO];
        descriptor.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> image = [c texture:descriptor];
        const float texels[4] = {0.5f, 1.5f, 2.5f, 3.5f};
        [image replaceRegion:MTLRegionMake2D(0, 0, 4, 1) mipmapLevel:0 withBytes:texels bytesPerRow:16];

        struct {
            uint64_t data;
            MTLResourceID image;
            uint32_t bias;
            uint32_t pad;
        } arguments = {data.gpuAddress, image.gpuResourceID, 11, 0};
        id<MTLBuffer> argumentBuffer = [c bufferWithBytes:&arguments length:sizeof arguments];
        id<MTLBuffer> out = [c buffer:8 * sizeof(uint32_t)];
        [c compute:^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pipeline];
            [e setBuffer:argumentBuffer offset:0 atIndex:0];
            [e setBuffer:out offset:0 atIndex:1];
            [e useResource:data usage:MTLResourceUsageRead];
            [e useResource:image usage:MTLResourceUsageRead];
            [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
        }];
        [c data:@"results" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];
    }
}
