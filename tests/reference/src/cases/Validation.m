// Validation cases: one API misuse per case, run without and with the Metal
// debug layer. Each records what the call returned (nil-ness, NSError
// domain and code, Objective-C exceptions); the runner records whether the
// process continued, exited or ended on a signal, with the assertion text.
// A step named after each call precedes it and a last step "returned"
// follows the misuse, so a run that ends inside a call shows which one.
//
// Misuse that the GPU would execute is only encoded: the command buffer is
// never committed, so no invalid work reaches the GPU in either variant.

#import "WMRefCase.h"
#import "LibrariesSupport.h"

static WMRule *WMOutcomeRule(void) {
    return WMNone(@"the result of invalid use is undefined in Metal unless the call documents an error; "
                  @"recorded for the defined-error table");
}

static MTLTextureDescriptor *WMTexture2D(MTLPixelFormat format, NSUInteger width, NSUInteger height) {
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width
                                                                                         height:height mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead;
    return descriptor;
}

#pragma mark - Resource creation

WMREF_CASE(validationBufferZero, "validation.buffer-length-zero", "validation.resources",
           "newBufferWithLength:0")
{
    __block id<MTLBuffer> buffer = nil;
    [c step:@"call"];
    [c call:@"call" rule:WMExact() block:^{
        buffer = [c.device newBufferWithLength:0 options:MTLResourceStorageModeShared];
    }];
    [c value:@"buffer-nil" boolean:buffer == nil rule:WMExact()];
    [c step:@"returned"];
}

WMREF_CASE(validationBufferOverMax, "validation.buffer-length-over-max", "validation.resources",
           "newBufferWithLength: one byte above maxBufferLength")
{
    NSUInteger length = c.device.maxBufferLength + 1;
    __block id<MTLBuffer> buffer = nil;
    [c step:@"call"];
    [c call:@"call" rule:WMExact() block:^{
        buffer = [c.device newBufferWithLength:length options:MTLResourceStorageModePrivate];
    }];
    [c value:@"buffer-nil" boolean:buffer == nil rule:WMExact()];
    [c step:@"returned"];
}

WMREF_CASE(validationTextureZeroWidth, "validation.texture-width-zero", "validation.resources",
           "newTextureWithDescriptor: with width 0")
{
    __block id<MTLTexture> texture = nil;
    [c step:@"call"];
    [c call:@"call" rule:WMExact() block:^{
        texture = [c.device newTextureWithDescriptor:WMTexture2D(MTLPixelFormatRGBA8Unorm, 0, 4)];
    }];
    [c value:@"texture-nil" boolean:texture == nil rule:WMExact()];
    [c step:@"returned"];
}

WMREF_CASE(validationTextureOverMax, "validation.texture-width-over-max", "validation.resources",
           "newTextureWithDescriptor: with a 2D width of 65536")
{
    // The Metal feature set tables give 16384 as the largest 2D texture
    // width of any Apple GPU family.
    __block id<MTLTexture> texture = nil;
    [c step:@"call"];
    [c call:@"call" rule:WMExact() block:^{
        texture = [c.device newTextureWithDescriptor:WMTexture2D(MTLPixelFormatR8Unorm, 65536, 1)];
    }];
    [c value:@"texture-nil" boolean:texture == nil rule:WMExact()];
    [c step:@"returned"];
}

WMREF_CASE(validationTextureMipCount, "validation.texture-mipmap-count-over-max", "validation.resources",
           "newTextureWithDescriptor: with more mipmap levels than the size allows")
{
    MTLTextureDescriptor *descriptor = WMTexture2D(MTLPixelFormatRGBA8Unorm, 64, 64);
    descriptor.mipmapLevelCount = 8; // a 64x64 texture has 7 levels
    __block id<MTLTexture> texture = nil;
    [c step:@"call"];
    [c call:@"call" rule:WMExact() block:^{
        texture = [c.device newTextureWithDescriptor:descriptor];
    }];
    [c value:@"texture-nil" boolean:texture == nil rule:WMExact()];
    [c step:@"returned"];
}

WMREF_CASE(validationTextureMultisampleMips, "validation.texture-multisample-with-mipmaps", "validation.resources",
           "newTextureWithDescriptor: for a 4-sample texture with two mipmap levels")
{
    MTLTextureDescriptor *descriptor = WMTexture2D(MTLPixelFormatRGBA8Unorm, 16, 16);
    descriptor.textureType = MTLTextureType2DMultisample;
    descriptor.sampleCount = 4;
    descriptor.mipmapLevelCount = 2;
    descriptor.usage = MTLTextureUsageRenderTarget;
    __block id<MTLTexture> texture = nil;
    [c step:@"call"];
    [c call:@"call" rule:WMExact() block:^{
        texture = [c.device newTextureWithDescriptor:descriptor];
    }];
    [c value:@"texture-nil" boolean:texture == nil rule:WMExact()];
    [c step:@"returned"];
}

WMREF_CASE(validationTextureSampleCount, "validation.texture-unsupported-sample-count", "validation.resources",
           "newTextureWithDescriptor: with a sample count the device does not support")
{
    NSUInteger count = 0;
    for (NSUInteger candidate = 3; candidate <= 8 && !count; candidate++)
        if (![c.device supportsTextureSampleCount:candidate]) count = candidate;
    [c require:@"unsupported-sample-count-exists" value:count != 0];
    [c value:@"sample-count" u64:count rule:WMDevice()];
    MTLTextureDescriptor *descriptor = WMTexture2D(MTLPixelFormatRGBA8Unorm, 16, 16);
    descriptor.textureType = MTLTextureType2DMultisample;
    descriptor.sampleCount = count;
    descriptor.usage = MTLTextureUsageRenderTarget;
    __block id<MTLTexture> texture = nil;
    [c step:@"call"];
    [c call:@"call" rule:WMExact() block:^{
        texture = [c.device newTextureWithDescriptor:descriptor];
    }];
    [c value:@"texture-nil" boolean:texture == nil rule:WMExact()];
    [c step:@"returned"];
}

WMREF_CASE(validationSamplerAnisotropy, "validation.sampler-anisotropy-zero", "validation.resources",
           "newSamplerStateWithDescriptor: with maxAnisotropy 0")
{
    // MTLSampler.h: the default is 1; the Metal documentation gives the
    // range 1 to 16.
    MTLSamplerDescriptor *descriptor = [MTLSamplerDescriptor new];
    descriptor.maxAnisotropy = 0;
    __block id<MTLSamplerState> sampler = nil;
    [c step:@"call"];
    [c call:@"call" rule:WMExact() block:^{
        sampler = [c.device newSamplerStateWithDescriptor:descriptor];
    }];
    [c value:@"sampler-nil" boolean:sampler == nil rule:WMExact()];
    [c step:@"returned"];
}

WMREF_CASE(validationTextureViewFormat, "validation.texture-view-without-format-view-usage", "validation.resources",
           "newTextureViewWithPixelFormat: to a different size-compatible format without the pixel-format-view usage")
{
    MTLTextureDescriptor *descriptor = WMTexture2D(MTLPixelFormatRGBA8Unorm, 4, 4);
    id<MTLTexture> texture = [c texture:descriptor];
    __block id<MTLTexture> view = nil;
    [c step:@"call"];
    [c call:@"call" rule:WMExact() block:^{
        view = [texture newTextureViewWithPixelFormat:MTLPixelFormatR32Float];
    }];
    [c value:@"view-nil" boolean:view == nil rule:WMExact()];
    [c step:@"returned"];
}

#pragma mark - Functions and pipelines

WMREF_CASE(validationComputeFromVertex, "validation.compute-pipeline-from-vertex-function", "validation.pipelines",
           "newComputePipelineStateWithFunction:error: with a vertex function")
{
    id<MTLFunction> vertex = [c function:@"pip_simple_vertex"];
    __block id<MTLComputePipelineState> state = nil;
    __block NSError *error = nil;
    [c step:@"call"];
    [c call:@"call" rule:WMExact() block:^{
        NSError *e = nil;
        state = [c.device newComputePipelineStateWithFunction:vertex error:&e];
        error = e;
    }];
    [c value:@"state-nil" boolean:state == nil rule:WMExact()];
    [c error:@"error" value:WMRefStableError(error) rule:WMRefDriverErrorRule()];
    [c step:@"returned"];
}

WMREF_CASE(validationRenderWithoutVertex, "validation.render-pipeline-without-vertex-function",
           "validation.pipelines", "newRenderPipelineStateWithDescriptor:error: with only a fragment function")
{
    MTLRenderPipelineDescriptor *descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.fragmentFunction = [c function:@"pip_simple_fragment"];
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    __block id<MTLRenderPipelineState> state = nil;
    __block NSError *error = nil;
    [c step:@"call"];
    [c call:@"call" rule:WMExact() block:^{
        NSError *e = nil;
        state = [c.device newRenderPipelineStateWithDescriptor:descriptor error:&e];
        error = e;
    }];
    [c value:@"state-nil" boolean:state == nil rule:WMExact()];
    [c error:@"error" value:WMRefStableError(error) rule:WMRefDriverErrorRule()];
    [c step:@"returned"];
}

WMREF_CASE(validationBlendInteger, "validation.render-pipeline-blending-integer-format", "validation.pipelines",
           "newRenderPipelineStateWithDescriptor:error: with blending enabled on an integer color format")
{
    MTLRenderPipelineDescriptor *descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [c function:@"pip_simple_vertex"];
    descriptor.fragmentFunction = [c function:@"pip_uint_fragment"];
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Uint;
    descriptor.colorAttachments[0].blendingEnabled = YES;
    __block id<MTLRenderPipelineState> state = nil;
    __block NSError *error = nil;
    [c step:@"call"];
    [c call:@"call" rule:WMExact() block:^{
        NSError *e = nil;
        state = [c.device newRenderPipelineStateWithDescriptor:descriptor error:&e];
        error = e;
    }];
    [c value:@"state-nil" boolean:state == nil rule:WMExact()];
    [c error:@"error" value:WMRefStableError(error) rule:WMRefDriverErrorRule()];
    [c step:@"returned"];
}

WMREF_CASE(validationConstantTypeMismatch, "validation.function-constant-type-mismatch", "validation.pipelines",
           "newFunctionWithName:constantValues:error: with a constant value of the wrong type")
{
    MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
    const float wrong = 2.0f;
    const bool flag = false;
    const float scale = 1.0f;
    const uint16_t half = 0x3c00, named = 1;
    const int32_t vector[2] = {0, 0};
    [values setConstantValue:&wrong type:MTLDataTypeFloat atIndex:0]; // declared uint
    [values setConstantValue:&flag type:MTLDataTypeBool atIndex:1];
    [values setConstantValue:&scale type:MTLDataTypeFloat atIndex:2];
    [values setConstantValue:&half type:MTLDataTypeHalf atIndex:4];
    [values setConstantValue:vector type:MTLDataTypeInt2 atIndex:5];
    [values setConstantValue:&named type:MTLDataTypeUShort atIndex:6];
    __block id<MTLFunction> function = nil;
    __block NSError *error = nil;
    [c step:@"call"];
    [c call:@"call" rule:WMExact() block:^{
        NSError *e = nil;
        function = [c.library newFunctionWithName:@"pip_constants" constantValues:values error:&e];
        error = e;
    }];
    [c value:@"function-nil" boolean:function == nil rule:WMExact()];
    [c error:@"error" value:WMRefStableError(error) rule:WMExact()];
    [c step:@"returned"];
}

#pragma mark - Encoding

static id<MTLComputePipelineState> WMMarkerPipeline(WMCase *c) {
    return [c computePipeline:@"pip_marker"];
}

WMREF_CASE(validationThreadgroupTooLarge, "validation.threadgroup-above-pipeline-limit", "validation.encoding",
           "dispatchThreadgroups with more threads per threadgroup than the pipeline allows (encoded, not committed)")
{
    id<MTLComputePipelineState> pipeline = WMMarkerPipeline(c);
    id<MTLBuffer> out = [c buffer:4096 * sizeof(uint32_t)];
    NSUInteger limit = pipeline.maxTotalThreadsPerThreadgroup;
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLComputeCommandEncoder> e = [commandBuffer computeCommandEncoder];
    [e setComputePipelineState:pipeline];
    [e setBuffer:out offset:0 atIndex:0];
    [c step:@"dispatch"];
    [c call:@"dispatch" rule:WMExact() block:^{
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(limit, 2, 1)];
    }];
    [e endEncoding];
    [c step:@"returned"];
}

WMREF_CASE(validationBufferIndex, "validation.set-buffer-index-over-limit", "validation.encoding",
           "setBuffer:offset:atIndex: at index 31 of a compute encoder (encoded, not committed)")
{
    // The buffer argument table has 31 entries (indices 0 to 30) in the
    // Metal feature set tables.
    id<MTLBuffer> out = [c buffer:64];
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLComputeCommandEncoder> e = [commandBuffer computeCommandEncoder];
    [c step:@"set-buffer"];
    [c call:@"set-buffer" rule:WMExact() block:^{
        [e setBuffer:out offset:0 atIndex:31];
    }];
    [e endEncoding];
    [c step:@"returned"];
}

WMREF_CASE(validationSetBytesTooLong, "validation.set-bytes-over-4k", "validation.encoding",
           "setBytes:length:atIndex: with 8 KiB of data (encoded, not committed)")
{
    // The Metal documentation limits setBytes data to 4 KiB.
    static uint8_t bytes[8192];
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLComputeCommandEncoder> e = [commandBuffer computeCommandEncoder];
    [c step:@"set-bytes"];
    [c call:@"set-bytes" rule:WMExact() block:^{
        [e setBytes:bytes length:sizeof bytes atIndex:0];
    }];
    [e endEncoding];
    [c step:@"returned"];
}

WMREF_CASE(validationConstantOffset, "validation.constant-buffer-offset-misaligned", "validation.encoding",
           "A constant float4 argument bound at offset 4 and dispatched (encoded, not committed)")
{
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"pip_constant_vector"];
    id<MTLBuffer> input = [c buffer:64];
    id<MTLBuffer> out = [c buffer:64];
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLComputeCommandEncoder> e = [commandBuffer computeCommandEncoder];
    [e setComputePipelineState:pipeline];
    [e setBuffer:out offset:0 atIndex:1];
    [c step:@"set-buffer"];
    [c call:@"set-buffer" rule:WMExact() block:^{
        [e setBuffer:input offset:4 atIndex:0];
    }];
    [c step:@"dispatch"];
    [c call:@"dispatch" rule:WMExact() block:^{
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }];
    [e endEncoding];
    [c step:@"returned"];
}

WMREF_CASE(validationDispatchWithoutPipeline, "validation.dispatch-without-pipeline", "validation.encoding",
           "dispatchThreadgroups without a compute pipeline state (encoded, not committed)")
{
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLComputeCommandEncoder> e = [commandBuffer computeCommandEncoder];
    [c step:@"dispatch"];
    [c call:@"dispatch" rule:WMExact() block:^{
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }];
    [e endEncoding];
    [c step:@"returned"];
}

static MTLRenderPassDescriptor *WMSmallPass(WMCase *c, MTLTextureUsage usage) {
    MTLTextureDescriptor *descriptor = WMTexture2D(MTLPixelFormatRGBA8Unorm, 4, 4);
    descriptor.usage = usage;
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = [c texture:descriptor];
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    return pass;
}

WMREF_CASE(validationDrawWithoutPipeline, "validation.draw-without-pipeline", "validation.encoding",
           "drawPrimitives without a render pipeline state (encoded, not committed)")
{
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLRenderCommandEncoder> e =
        [commandBuffer renderCommandEncoderWithDescriptor:WMSmallPass(c, MTLTextureUsageRenderTarget)];
    [c step:@"draw"];
    [c call:@"draw" rule:WMExact() block:^{
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }];
    [e endEncoding];
    [c step:@"returned"];
}

WMREF_CASE(validationRenderTargetUsage, "validation.render-target-without-usage", "validation.encoding",
           "renderCommandEncoderWithDescriptor: with a color texture that lacks the render-target usage")
{
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    MTLRenderPassDescriptor *pass = WMSmallPass(c, MTLTextureUsageShaderRead);
    __block id<MTLRenderCommandEncoder> e = nil;
    [c step:@"create-encoder"];
    [c call:@"create-encoder" rule:WMExact() block:^{
        e = [commandBuffer renderCommandEncoderWithDescriptor:pass];
    }];
    [c value:@"encoder-nil" boolean:e == nil rule:WMExact()];
    [e endEncoding];
    [c step:@"returned"];
}

WMREF_CASE(validationTextureWriteUsage, "validation.texture-write-without-shader-write", "validation.encoding",
           "A kernel that writes a texture created without the shader-write usage (encoded, not committed)")
{
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"pip_write_texture"];
    id<MTLTexture> texture = [c texture:WMTexture2D(MTLPixelFormatRGBA8Unorm, 4, 4)];
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLComputeCommandEncoder> e = [commandBuffer computeCommandEncoder];
    [e setComputePipelineState:pipeline];
    [c step:@"set-texture"];
    [c call:@"set-texture" rule:WMExact() block:^{
        [e setTexture:texture atIndex:0];
    }];
    [c step:@"dispatch"];
    [c call:@"dispatch" rule:WMExact() block:^{
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 4, 1)];
    }];
    [e endEncoding];
    [c step:@"returned"];
}

WMREF_CASE(validationUseAfterEnd, "validation.encoder-use-after-end-encoding", "validation.encoding",
           "setComputePipelineState: on a compute encoder after endEncoding")
{
    id<MTLComputePipelineState> pipeline = WMMarkerPipeline(c);
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLComputeCommandEncoder> e = [commandBuffer computeCommandEncoder];
    [e endEncoding];
    [c step:@"set-pipeline-after-end"];
    [c call:@"set-pipeline-after-end" rule:WMExact() block:^{
        [e setComputePipelineState:pipeline];
    }];
    [c step:@"returned"];
}

WMREF_CASE(validationEncoderAfterCommit, "validation.encoder-on-committed-buffer", "validation.encoding",
           "computeCommandEncoder on a command buffer that was committed")
{
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    [c value:@"status" i64:commandBuffer.status rule:WMExact()];
    __block id<MTLComputeCommandEncoder> e = nil;
    [c step:@"create-encoder"];
    [c call:@"create-encoder" rule:WMExact() block:^{
        e = [commandBuffer computeCommandEncoder];
    }];
    [c value:@"encoder-nil" boolean:e == nil rule:WMExact()];
    [c step:@"returned"];
}

WMREF_CASE(validationTwoEncoders, "validation.second-encoder-while-open", "validation.encoding",
           "blitCommandEncoder while a compute encoder of the same command buffer is still open")
{
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLComputeCommandEncoder> compute = [commandBuffer computeCommandEncoder];
    __block id<MTLBlitCommandEncoder> blit = nil;
    [c step:@"create-second-encoder"];
    [c call:@"create-second-encoder" rule:WMExact() block:^{
        blit = [commandBuffer blitCommandEncoder];
    }];
    [c value:@"encoder-nil" boolean:blit == nil rule:WMExact()];
    [c step:@"returned"];
    [blit endEncoding];
    [compute endEncoding];
}

#pragma mark - CPU access to private resources

WMREF_CASE(validationPrivateContents, "validation.private-buffer-contents", "validation.cpu-access",
           "contents of a private-storage buffer")
{
    id<MTLBuffer> buffer = [c.device newBufferWithLength:256 options:MTLResourceStorageModePrivate];
    if (!buffer) [c fail:@"private buffer allocation failed"];
    __block void *contents = (void *)1;
    [c step:@"call"];
    [c call:@"call" rule:WMExact() block:^{
        contents = buffer.contents;
    }];
    [c value:@"contents-null" boolean:contents == NULL rule:WMOutcomeRule()];
    [c step:@"returned"];
}

WMREF_CASE(validationPrivateGetBytes, "validation.private-texture-get-bytes", "validation.cpu-access",
           "getBytes:bytesPerRow:fromRegion:mipmapLevel: on a private-storage texture")
{
    id<MTLTexture> texture = [c texture:WMTexture2D(MTLPixelFormatRGBA8Unorm, 4, 4)];
    static uint8_t bytes[64];
    [c step:@"call"];
    [c call:@"call" rule:WMExact() block:^{
        [texture getBytes:bytes bytesPerRow:16 fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    }];
    [c step:@"returned"];
}

WMREF_CASE(validationPrivateReplace, "validation.private-texture-replace-region", "validation.cpu-access",
           "replaceRegion:mipmapLevel:withBytes:bytesPerRow: on a private-storage texture")
{
    id<MTLTexture> texture = [c texture:WMTexture2D(MTLPixelFormatRGBA8Unorm, 4, 4)];
    static uint8_t bytes[64];
    [c step:@"call"];
    [c call:@"call" rule:WMExact() block:^{
        [texture replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:bytes bytesPerRow:16];
    }];
    [c step:@"returned"];
}

WMREF_CASE(validationPrivateInitial, "validation.private-texture-initial-contents", "validation.undefined",
           "Contents of a new private-storage texture before anything writes it")
{
    MTLTextureDescriptor *descriptor = WMTexture2D(MTLPixelFormatRGBA8Unorm, 8, 8);
    id<MTLTexture> texture = [c texture:descriptor];
    NSData *bytes = [c readTexture:texture level:0 slice:0 bytesPerRow:32];
    BOOL zero = YES;
    for (NSUInteger i = 0; i < bytes.length; i++) zero &= ((const uint8_t *)bytes.bytes)[i] == 0;
    [c value:@"all-zero" boolean:zero
        rule:WMNone(@"Metal does not define the initial contents of a new private texture")];
}
