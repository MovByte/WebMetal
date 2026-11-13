// Pipeline SPI answers (AGXA10FamilyComputePipeline and AGXA10FamilyRenderPipeline on 21H461): the constants the A10
// family's pipelines answer, since the family has no imageblocks (supportsImageBlocks is NO), mesh or object
// functions, or pipelines in resource tables.

#import "WebMetalSPI.h"

// The class adopts the driver's protocols; these categories implement the driver selectors they declare.
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"

@implementation WebMetalComputePipelineState (WebMetalSPI)

// -[AGXA10FamilyComputePipeline imageblockMemoryLengthForDimensions:] (AGXMetalA10 0x21c8cc3c8): the constant 0.
- (NSUInteger)imageblockMemoryLengthForDimensions:(MTLSize)dimensions
{
    return 0;
}

// -[AGXA10FamilyComputePipeline imageBlockMemoryLengthForDimensions:] (AGXMetalA10 0x21c8cc3d0): the constant 0.
- (NSUInteger)imageBlockMemoryLengthForDimensions:(MTLSize)dimensions
{
    return 0;
}

// -[AGXA10FamilyComputePipeline resourceIndex] (AGXMetalA10 0x21c8cc7c8): the constant 0.
- (NSUInteger)resourceIndex
{
    return 0;
}

@end

@implementation WebMetalRenderPipelineState (WebMetalSPI)

// -[AGXA10FamilyRenderPipeline imageblockSampleLength] (AGXMetalA10 0x21c977a74): the constant 0.
- (NSUInteger)imageblockSampleLength
{
    return 0;
}

// -[AGXA10FamilyRenderPipeline imageBlockSampleLength] (AGXMetalA10 0x21c977a5c): the constant 0.
- (NSUInteger)imageBlockSampleLength
{
    return 0;
}

// -[AGXA10FamilyRenderPipeline imageblockMemoryLengthForDimensions:] (AGXMetalA10 0x21c977a6c): the constant 0.
- (NSUInteger)imageblockMemoryLengthForDimensions:(MTLSize)dimensions
{
    return 0;
}

// -[AGXA10FamilyRenderPipeline imageBlockMemoryLengthForDimensions:] (AGXMetalA10 0x21c977a64): the constant 0.
- (NSUInteger)imageBlockMemoryLengthForDimensions:(MTLSize)dimensions
{
    return 0;
}

// -[AGXA10FamilyRenderPipeline staticThreadgroupMemoryLength] (AGXMetalA10 0x21c977a54): the constant 0.
- (NSUInteger)staticThreadgroupMemoryLength
{
    return 0;
}

// -[AGXA10FamilyRenderPipeline objectThreadExecutionWidth] (AGXMetalA10 0x21c977c18): the constant 32.
- (NSUInteger)objectThreadExecutionWidth
{
    return 32;
}

// -[AGXA10FamilyRenderPipeline meshThreadExecutionWidth] (AGXMetalA10 0x21c977c20): the constant 32.
- (NSUInteger)meshThreadExecutionWidth
{
    return 32;
}

// -[AGXA10FamilyRenderPipeline resourceIndex] (AGXMetalA10 0x21c977a98): the constant 0.
- (NSUInteger)resourceIndex
{
    return 0;
}

@end
