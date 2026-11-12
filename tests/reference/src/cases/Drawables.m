// Drawable cases: CAMetalLayer defaults, drawables obtained from a layer
// that is not attached to a window, rendering into a drawable and
// presenting it from a command buffer, and the drawable pool limit.
// Every wait is bounded; nothing here needs a display.

#import "WMRefCase.h"
#import "LibrariesSupport.h"

#import <QuartzCore/QuartzCore.h>

static CAMetalLayer *WMLayer(WMCase *c, BOOL framebufferOnly) {
    CAMetalLayer *layer = [CAMetalLayer layer];
    layer.device = c.device;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.drawableSize = CGSizeMake(16, 8);
    layer.framebufferOnly = framebufferOnly;
    return layer;
}

WMREF_CASE(drawableLayerDefaults, "drawable.layer-defaults", "drawable.layer",
           "CAMetalLayer default properties and the maximumDrawableCount range")
{
    CAMetalLayer *layer = [CAMetalLayer layer];
    // Stated in CAMetalLayer.h.
    [c value:@"maximum-drawable-count" u64:layer.maximumDrawableCount rule:WMExact()];
    [c value:@"presents-with-transaction" boolean:layer.presentsWithTransaction rule:WMExact()];
    [c value:@"allows-next-drawable-timeout" boolean:layer.allowsNextDrawableTimeout rule:WMExact()];
    [c value:@"opaque" boolean:layer.opaque rule:WMExact()];
#if TARGET_OS_OSX
    [c value:@"display-sync-enabled" boolean:layer.displaySyncEnabled rule:WMExact()];
    // macOS: the device defaults to nil (CAMetalLayer.h).
    [c value:@"device-nil" boolean:layer.device == nil rule:WMExact()];
#endif
    WMRule *unstated = WMNone(@"default not stated in CAMetalLayer.h");
    [c value:@"pixel-format" u64:layer.pixelFormat rule:unstated];
    [c value:@"framebuffer-only" boolean:layer.framebufferOnly rule:unstated];

    // Values outside [2, 3] are ignored and an exception is thrown
    // (CAMetalLayer.h).
    [c call:@"set-maximum-drawable-count-4" rule:WMExact() block:^{
        layer.maximumDrawableCount = 4;
    }];
    [c value:@"maximum-drawable-count-after-4" u64:layer.maximumDrawableCount rule:WMExact()];
    [c call:@"set-maximum-drawable-count-2" rule:WMExact() block:^{
        layer.maximumDrawableCount = 2;
    }];
    [c value:@"maximum-drawable-count-after-2" u64:layer.maximumDrawableCount rule:WMExact()];
}

WMREF_CASE(drawableRenderPresent, "drawable.render-present", "drawable.present",
           "Render into a drawable of a layer without a window, read it back and present it")
{
    CAMetalLayer *layer = WMLayer(c, NO);
    id<CAMetalDrawable> drawable = layer.nextDrawable;
    [c require:@"next-drawable" value:drawable != nil];
    id<MTLTexture> texture = drawable.texture;
    [c value:@"texture.width" u64:texture.width rule:WMExact()];
    [c value:@"texture.height" u64:texture.height rule:WMExact()];
    [c value:@"texture.pixel-format" u64:texture.pixelFormat rule:WMExact()];
    [c value:@"texture.type" u64:texture.textureType rule:WMExact()];
    [c value:@"texture.framebuffer-only" boolean:texture.framebufferOnly rule:WMExact()];
    [c value:@"texture.usage" u64:texture.usage rule:WMNone(@"usage beyond the framebufferOnly contract is chosen by Core Animation")];
    [c value:@"texture.storage-mode" u64:texture.storageMode rule:WMDevice()];
    [c value:@"texture.same-device" boolean:texture.device == c.device rule:WMExact()];
    [c value:@"texture.iosurface-backed" boolean:texture.iosurface != NULL
        rule:WMNone(@"the backing of drawable textures is chosen by Core Animation")];
    [c value:@"drawable.layer-matches" boolean:drawable.layer == layer rule:WMExact()];

    __block BOOL presented = NO;
    dispatch_semaphore_t presentedSignal = dispatch_semaphore_create(0);
    [drawable addPresentedHandler:^(id<MTLDrawable> d) {
        presented = YES;
        dispatch_semaphore_signal(presentedSignal);
    }];

    // Clear to values exactly representable in BGRA8Unorm, copy to a
    // buffer, then present.
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 1, 1);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLBuffer> pixels = [c buffer:16 * 8 * 4];
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLRenderCommandEncoder> render = [commandBuffer renderCommandEncoderWithDescriptor:pass];
    [render endEncoding];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(16, 8, 1) toBuffer:pixels destinationOffset:0 destinationBytesPerRow:64
    destinationBytesPerImage:512];
    [blit endEncoding];
    [commandBuffer presentDrawable:drawable];
    [c step:@"commit"];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    [c value:@"command-buffer.status" i64:commandBuffer.status rule:WMExact()];
    [c error:@"command-buffer.error" value:commandBuffer.error rule:WMExact()];
    [c data:@"pixels" elem:WMElemU8 bytes:pixels.contents length:pixels.length
     layout:@{@"format": @"BGRA8Unorm", @"width": @16, @"height": @8, @"depth": @1, @"bytes_per_row": @64}
       rule:WMExact()];
    BOOL called = WMRefWait(presentedSignal, 2.0);
    [c value:@"presented-handler-within-2s" boolean:called && presented
        rule:WMNone(@"presentation of a layer without a window or display depends on the window server")];
}

WMREF_CASE(drawableFramebufferOnly, "drawable.framebuffer-only", "drawable.layer",
           "Texture properties of a drawable from a framebufferOnly layer")
{
    CAMetalLayer *layer = WMLayer(c, YES);
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
    id<CAMetalDrawable> drawable = layer.nextDrawable;
    [c require:@"next-drawable" value:drawable != nil];
    [c value:@"texture.framebuffer-only" boolean:drawable.texture.framebufferOnly rule:WMExact()];
    [c value:@"texture.pixel-format" u64:drawable.texture.pixelFormat rule:WMExact()];
    [c value:@"texture.render-target-usage"
     boolean:(drawable.texture.usage & MTLTextureUsageRenderTarget) != 0 rule:WMExact()];
}

WMREF_CASE(drawablePoolLimit, "drawable.pool-exhaustion", "drawable.layer",
           "nextDrawable when every drawable of the pool is held and not presented")
{
    CAMetalLayer *layer = WMLayer(c, YES);
    layer.maximumDrawableCount = 2;
    NSMutableArray *held = [NSMutableArray array];
    for (int i = 0; i < 2; i++) {
        id<CAMetalDrawable> drawable = layer.nextDrawable;
        [c value:[NSString stringWithFormat:@"drawable-%d-nil", i] boolean:drawable == nil rule:WMExact()];
        if (drawable) [held addObject:drawable];
    }
    // CAMetalLayer.h lists "all drawables in the swap queue are in-use and
    // the 1 second timeout has elapsed" as a condition for nil. Here both
    // drawables were obtained and never presented; the result is recorded
    // as measured.
    NSDate *start = [NSDate date];
    id<CAMetalDrawable> extra = layer.nextDrawable;
    NSTimeInterval elapsed = -start.timeIntervalSinceNow;
    [c value:@"extra-nil" boolean:extra == nil rule:WMExact()];
    [c value:@"extra-waited-at-least-0.9s" boolean:elapsed >= 0.9
        rule:WMNone(@"timing; CAMetalLayer.h states a 1 second timeout")];
    [held removeAllObjects];
}
