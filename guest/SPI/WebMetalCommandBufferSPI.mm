// Command-buffer SPI (AGXA10FamilyCommandBuffer on 21H461): encoder commits and cache hints.

#import "WebMetalSPI.h"
#import "../Commands/WebMetalCommandState.h"

// The class adopts the driver's protocols; these categories implement the driver selectors they declare.
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"

@implementation WebMetalCommandBuffer (WebMetalSPI)

// -[AGXA10FamilyCommandBuffer commitEncoder] (AGXMetalA10 0x21c8c8fac) ends the encoding the driver deferred: the
// blit, compute and render encoders whose endEncoding it postponed to coalesce them with the next encoder of their
// kind (-deferredEndEncoding, -deferredEndEncoding:). The plugin's encoders end at -endEncoding, so none is deferred.
- (void)commitEncoder
{
}

// -[AGXA10FamilyCommandBuffer encodeCacheHintTag:resourceGroups:count:] (AGXMetalA10 0x21c8c91a8) returns at once.
- (void)encodeCacheHintTag:(NSUInteger)tag resourceGroups:(const id __unsafe_unretained *)groups count:(NSUInteger)count
{
}

// -[AGXA10FamilyCommandBuffer encodeCacheHintFinalize:resourceGroups:count:] (AGXMetalA10 0x21c8c91ac): report an open
// encoder, commit the deferred encoders, and for hint 1 encode a kernel cache command for each resource group of the
// list that is not nil. No resource group exists on the A10 family: -[AGXA10FamilyDevice
// newResourceGroupFromResources:count:] answers nil (guest/Resources/WebMetalDeviceResources.mm), so the list holds only
// nils and the driver encodes nothing.
- (void)encodeCacheHintFinalize:(NSUInteger)hint
                 resourceGroups:(const id __unsafe_unretained *)groups
                          count:(NSUInteger)count
{
    if (_currentEncoder)
        MTLReportFailure(0, "-[AGXA10FamilyCommandBuffer encodeCacheHintFinalize:resourceGroups:count:]", 1585,
                         @"A command encoder is already encoding to this command buffer");
    [self commitEncoder];
    if (hint != 1)
        return;
    for (NSUInteger i = 0; i < count; i++)
        if (groups[i])
            WMLoudMethod(self, _cmd, "an object in place of a resource group, which no A10-family device creates");
}

// -[AGXA10FamilyCommandBuffer setResourceGroups:count:] and -encodeDashboardTagForResourceGroup: (AGXMetalA10
// 0x21c8c91a4, 0x21c8c92ac) return at once.
- (void)setResourceGroups:(const id __unsafe_unretained *)groups count:(NSUInteger)count
{
}

- (void)encodeDashboardTagForResourceGroup:(id)group
{
}

@end
