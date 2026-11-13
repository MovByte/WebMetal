// Texture SPI (AGXA10FamilyTexture on 21H461) whose A10 implementations answer a constant: the family has no sparse
// textures and no lossless compression.

#import "WebMetalSPI.h"
#import "../Resources/WebMetalTexture.h"

// The class adopts the driver's protocols; these categories implement the driver selectors they declare.
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"

@implementation WebMetalTexture (WebMetalSPI)

// -[AGXA10FamilyTexture compressionFeedback] (AGXMetalA10 0x21c9823c4): the constant 4.
- (NSInteger)compressionFeedback
{
    return 4;
}

// -[AGXA10FamilyTexture firstMipmapInTail] (AGXMetalA10 0x21c9822f8): the constant 0.
- (NSUInteger)firstMipmapInTail
{
    return 0;
}

// -[AGXA10FamilyTexture tailSize] and -tailSizeInBytes (AGXMetalA10 0x21c9822e8, 0x21c9822f0): the constant 0.
- (NSUInteger)tailSize
{
    return 0;
}

- (NSUInteger)tailSizeInBytes
{
    return 0;
}

// -[AGXA10FamilyTexture sparseSurfaceDefaultValue] (AGXMetalA10 0x21c9822e0): the constant 0.
- (NSUInteger)sparseSurfaceDefaultValue
{
    return 0;
}

// -[AGXA10FamilyTexture setSparseHeap:] (AGXMetalA10 0x21c981ef4) returns at once.
- (void)setSparseHeap:(id)heap
{
}

@end
