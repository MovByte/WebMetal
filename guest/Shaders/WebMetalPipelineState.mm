// The pipeline state objects (AGXA10FamilyComputePipeline and AGXA10FamilyRenderPipeline on 21H461).
//
// The driver's pipelines hold the machine code its back end compiled and answer their limits from it. The plugin's
// pipelines hold the wire handle the host created the pipeline under; Metal.framework's base classes keep the
// descriptor state they keep on 21H461 (label, device, supportIndirectCommandBuffers...).

#import "WebMetalShaders.h"
#import "../Commands/WebMetalCommands.h"

#pragma clang diagnostic ignored "-Wprotocol"
#pragma clang diagnostic ignored "-Wobjc-protocol-property-synthesis"
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"

namespace {

// Why a value the driver takes from its back end's reply has no WebMetal source.
const char *const kBackEndValue =
    "the A10X driver answers it from its back-end compiler's reply (AGXCompilerCore's register and memory counts, read "
    "by AGX::ComputeProgramVariant's constructor at 0x21c92f9b4), which the host's compiler does not reproduce";

}  // namespace

@implementation WebMetalComputePipelineState

// -[AGXA10FamilyComputePipeline initWithDevice:pipelineStateDescriptor:] (AGXMetalA10 0x21c8cc26c) initializes
// _MTLComputePipelineState and keeps the compute function's driver program (its vendorPrivate), which has no
// counterpart here: the host holds the compiled pipeline.
- (instancetype)initWithDevice:(id<MTLDevice>)device pipelineStateDescriptor:(MTLComputePipelineDescriptor *)descriptor
{
    return [super initWithDevice:device pipelineStateDescriptor:descriptor];
}

- (void)dealloc
{
    if (_handle) {
        webmetal::ForgetHandle((__bridge const void *)self);
        webmetal::ReleaseHandle(_connection, _handle);
    }
}

// -[AGXA10FamilyComputePipeline threadExecutionWidth] (AGXMetalA10 0x21c8cc394): the constant 32, the A10X's SIMD
// group width.
- (NSUInteger)threadExecutionWidth
{
    return 32;
}

// -[AGXA10FamilyComputePipeline maxTotalThreadsPerThreadgroup] (AGXMetalA10 0x21c8cc39c): the variant's limit, which
// ComputeProgramVariant's constructor computes from the reply's register counts (0x21c934494-0x21c9345b0).
- (NSUInteger)maxTotalThreadsPerThreadgroup
{
    WMLoudMethod(self, _cmd, kBackEndValue);
}

// -[AGXA10FamilyComputePipeline staticThreadgroupMemoryLength] (AGXMetalA10 0x21c8cc3b0): the variant's threadgroup
// memory, the reply's value rounded up to 16 bytes (0x21c934458-0x21c934488).
- (NSUInteger)staticThreadgroupMemoryLength
{
    WMLoudMethod(self, _cmd, kBackEndValue);
}

// -[AGXA10FamilyComputePipeline allocatedSize] (AGXMetalA10 0x21c8cc7d0): the variant's code size, plus 232 bytes when
// the pipeline supports indirect command buffers.
- (NSUInteger)allocatedSize
{
    WMLoudMethod(self, _cmd, "the A10X driver answers the size of its back end's machine code, which has no WebMetal "
                             "counterpart");
}

@end

@implementation WebMetalRenderPipelineState

// -[AGXA10FamilyRenderPipeline initWithDevice:pipelineStateDescriptor:] (AGXMetalA10 0x21c9774cc) initializes
// _MTLRenderPipelineState and keeps the stage functions' driver programs (their vendorPrivate), which have no
// counterpart here.
- (instancetype)initWithDevice:(id<MTLDevice>)device pipelineStateDescriptor:(MTLRenderPipelineDescriptor *)descriptor
{
    return [super initWithDevice:device pipelineStateDescriptor:descriptor];
}

- (void)dealloc
{
    if (_handle) {
        webmetal::ForgetHandle((__bridge const void *)self);
        webmetal::ReleaseHandle(_connection, _handle);
    }
}

- (NSUInteger)allocatedSize
{
    WMLoudMethod(self, _cmd, "the A10X driver answers the size of its back end's machine code, which has no WebMetal "
                             "counterpart");
}

@end
