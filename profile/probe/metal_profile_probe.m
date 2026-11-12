// Second method of the device profile: run this Mac's own Metal.framework query code for a
// device whose driver answers are the A10X's.
//
// The probe subclasses the framework's private _MTLDevice at run time. Every selector that the
// target's driver classes (AGXA10XDevice, AGXA10FamilyDevice, IOGPUMetalDevice) implement is
// installed on the subclass: getters with the value method A extracted, every other driver
// method with a trap. The instance then goes through MTLAddDevice's post-init sequence and the
// framework's own implementations answer the queries. A query during which a trap fired is
// reported as unreachable together with the trapped selectors; its value is not used.
//
// Input: a JSON file (argv[1]). Output: JSON on stdout.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>

typedef struct { uint64_t a, b; } Pair;

static NSDictionary *gValues;
static NSMutableArray<NSString *> *gTrapped;

static id answer(SEL cmd) {
  id v = gValues[NSStringFromSelector(cmd)];
  if (v == nil) {
    fprintf(stderr, "probe: no answer for %s\n", sel_getName(cmd));
    _exit(4);
  }
  return v;
}

static id tableAnswer(SEL cmd, uint64_t arg) {
  NSDictionary *table = answer(cmd);
  id v = table[[NSString stringWithFormat:@"%llu", (unsigned long long)arg]];
  if (v == nil) {
    [gTrapped addObject:[NSString stringWithFormat:@"%s(%llu)", sel_getName(cmd), (unsigned long long)arg]];
    return @0;
  }
  return v;
}

static BOOL retB(id self, SEL cmd) { return [answer(cmd) boolValue]; }
static uint64_t retQ(id self, SEL cmd) { return [answer(cmd) unsignedLongLongValue]; }
static int64_t retq(id self, SEL cmd) { return [answer(cmd) longLongValue]; }
static uint32_t retI(id self, SEL cmd) { return [answer(cmd) unsignedIntValue]; }
static float retf(id self, SEL cmd) { return [answer(cmd) floatValue]; }
static MTLSize retSize(id self, SEL cmd) {
  NSArray *a = answer(cmd);
  return MTLSizeMake([a[0] unsignedLongLongValue], [a[1] unsignedLongLongValue], [a[2] unsignedLongLongValue]);
}
static BOOL tabB(id self, SEL cmd, uint64_t arg) { return [tableAnswer(cmd, arg) boolValue]; }
static uint64_t tabQ(id self, SEL cmd, uint64_t arg) { return [tableAnswer(cmd, arg) unsignedLongLongValue]; }
static Pair trap(id self, SEL cmd) {
  [gTrapped addObject:@(sel_getName(cmd))];
  Pair zero = {0, 0};
  return zero;
}

// A selector neither the host framework nor the target's driver classes implement: the target
// device would not respond to it either. It is trapped like a driver method, so the query that
// sent it is reported as unreachable instead of ending the probe.
static BOOL resolveUnknown(id cls, SEL cmd, SEL sel) {
  [gTrapped addObject:[NSString stringWithFormat:@"unimplemented %s", sel_getName(sel)]];
  class_addMethod(cls, sel, (IMP)trap, "v@:");
  return YES;
}

static NSArray *trappedAndReset(void) {
  NSArray *out = [gTrapped copy];
  [gTrapped removeAllObjects];
  return out;
}

static id result(id value) {
  NSArray *t = trappedAndReset();
  if (t.count) return @{@"trapped": t};
  return @{@"value": value ?: [NSNull null]};
}

static id callZeroArg(id dev, SEL sel, NSString *ret) {
  if ([ret isEqualToString:@"B"]) return @(((BOOL(*)(id, SEL))objc_msgSend)(dev, sel));
  if ([ret isEqualToString:@"Q"]) return @(((uint64_t(*)(id, SEL))objc_msgSend)(dev, sel));
  if ([ret isEqualToString:@"q"]) return @(((int64_t(*)(id, SEL))objc_msgSend)(dev, sel));
  if ([ret isEqualToString:@"I"]) return @(((uint32_t(*)(id, SEL))objc_msgSend)(dev, sel));
  if ([ret isEqualToString:@"f"]) return @(((float(*)(id, SEL))objc_msgSend)(dev, sel));
  if ([ret isEqualToString:@"size"]) {
    MTLSize s = ((MTLSize(*)(id, SEL))objc_msgSend)(dev, sel);
    return @[@(s.width), @(s.height), @(s.depth)];
  }
  if ([ret isEqualToString:@"pair"]) {
    Pair p = ((Pair(*)(id, SEL))objc_msgSend)(dev, sel);
    return @[@(p.a), @(p.b)];
  }
  return nil;
}

static id callOneArg(id dev, SEL sel, NSString *ret, uint64_t arg) {
  if ([ret isEqualToString:@"B"]) return @(((BOOL(*)(id, SEL, uint64_t))objc_msgSend)(dev, sel, arg));
  if ([ret isEqualToString:@"Q"]) return @(((uint64_t(*)(id, SEL, uint64_t))objc_msgSend)(dev, sel, arg));
  return nil;
}

static NSString *hex(const void *bytes, size_t n) {
  NSMutableString *s = [NSMutableString string];
  for (size_t i = 0; i < n; i++) [s appendFormat:@"%02x", ((const uint8_t *)bytes)[i]];
  return s;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) {
      fprintf(stderr, "usage: %s input.json\n", argv[0]);
      return 2;
    }
    NSData *data = [NSData dataWithContentsOfFile:@(argv[1])];
    NSDictionary *input = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    if (!input) {
      fprintf(stderr, "probe: unreadable input\n");
      return 2;
    }
    gTrapped = [NSMutableArray array];
    NSMutableDictionary *values = [NSMutableDictionary dictionary];
    Class base = objc_getClass("_MTLDevice");
    if (!base) {
      fprintf(stderr, "probe: no _MTLDevice\n");
      return 2;
    }
    Class probe = objc_allocateClassPair(base, "WebMetalProfileProbeDevice", 0);
    NSDictionary *driver = input[@"driver"];
    for (NSString *selector in driver) {
      NSDictionary *d = driver[selector];
      SEL sel = sel_registerName(selector.UTF8String);
      NSString *kind = d[@"kind"];
      IMP imp = (IMP)trap;
      if ([kind isEqualToString:@"value"]) {
        NSString *ret = d[@"ret"];
        values[selector] = d[@"value"];
        if ([ret isEqualToString:@"B"]) imp = (IMP)retB;
        else if ([ret isEqualToString:@"Q"]) imp = (IMP)retQ;
        else if ([ret isEqualToString:@"q"]) imp = (IMP)retq;
        else if ([ret isEqualToString:@"I"]) imp = (IMP)retI;
        else if ([ret isEqualToString:@"f"]) imp = (IMP)retf;
        else if ([ret isEqualToString:@"size"]) imp = (IMP)retSize;
        else {
          fprintf(stderr, "probe: return kind %s\n", ret.UTF8String);
          return 2;
        }
      } else if ([kind isEqualToString:@"table"]) {
        values[selector] = d[@"table"];
        imp = [d[@"ret"] isEqualToString:@"B"] ? (IMP)tabB : (IMP)tabQ;
      }
      class_addMethod(probe, sel, imp, [d[@"types"] UTF8String]);
    }
    gValues = values;
    objc_registerClassPair(probe);
    class_addMethod(object_getClass(probe), @selector(resolveInstanceMethod:), (IMP)resolveUnknown, "B@::");

    // The instance is never released: _MTLDevice's dealloc expects state that only a real
    // device initializer creates.
    id dev = (__bridge id)CFBridgingRetain(class_createInstance(probe, 0));
    NSMutableDictionary *out = [NSMutableDictionary dictionary];
    NSMutableArray *sequence = [NSMutableArray array];
    for (NSString *selector in input[@"sequence"]) {
      SEL sel = sel_registerName(selector.UTF8String);
      if (![dev respondsToSelector:sel]) {
        [sequence addObject:@{@"selector": selector, @"missing": @YES}];
        continue;
      }
      ((void (*)(id, SEL))objc_msgSend)(dev, sel);
      [sequence addObject:@{@"selector": selector, @"trapped": trappedAndReset()}];
    }
    out[@"sequence"] = sequence;

    Ivar limits = class_getInstanceVariable(base, "_limits");
    NSUInteger limitsSize = 0, limitsAlign = 0;
    NSGetSizeAndAlignment(ivar_getTypeEncoding(limits), &limitsSize, &limitsAlign);
    out[@"limits"] = @{
      @"encoding": @(ivar_getTypeEncoding(limits)),
      @"offset": @(ivar_getOffset(limits)),
      @"bytes": hex((uint8_t *)(__bridge void *)dev + ivar_getOffset(limits), limitsSize),
    };

    NSMutableDictionary *getters = [NSMutableDictionary dictionary];
    NSDictionary *wanted = input[@"getters"];
    for (NSString *selector in wanted) {
      SEL sel = sel_registerName(selector.UTF8String);
      if (![dev respondsToSelector:sel]) {
        getters[selector] = @{@"unimplemented": @YES};
        continue;
      }
      getters[selector] = result(callZeroArg(dev, sel, wanted[selector]));
    }
    out[@"getters"] = getters;

    NSMutableDictionary *parameterized = [NSMutableDictionary dictionary];
    NSDictionary *pwanted = input[@"parameterized"];
    for (NSString *selector in pwanted) {
      SEL sel = sel_registerName(selector.UTF8String);
      NSMutableDictionary *answers = [NSMutableDictionary dictionary];
      if (![dev respondsToSelector:sel]) {
        parameterized[selector] = @{@"unimplemented": @YES};
        continue;
      }
      for (NSNumber *arg in pwanted[selector][@"args"])
        answers[arg.stringValue] = result(callOneArg(dev, sel, pwanted[selector][@"ret"], arg.unsignedLongLongValue));
      parameterized[selector] = answers;
    }
    out[@"parameterized"] = parameterized;

    typedef struct { uint8_t bytes[0x38]; } Info;
    Info (*getInfo)(id, uint64_t) = (Info(*)(id, uint64_t))dlsym(RTLD_DEFAULT, "MTLPixelFormatGetInfoForDevice");
    NSMutableDictionary *formats = [NSMutableDictionary dictionary];
    if (getInfo) {
      for (NSNumber *f in input[@"pixel_formats"]) {
        Info info = getInfo(dev, f.unsignedLongLongValue);
        // The first word is a pointer to the format's name; it is not compared.
        formats[f.stringValue] = result(hex(info.bytes + 8, sizeof(info) - 8));
      }
    }
    out[@"pixel_formats"] = formats;

    NSMutableDictionary *positions = [NSMutableDictionary dictionary];
    SEL supports = sel_registerName("supportsTextureSampleCount:");
    SEL getPositions = sel_registerName("getDefaultSamplePositions:count:");
    for (NSNumber *n in input[@"sample_counts"]) {
      NSUInteger count = n.unsignedIntegerValue;
      BOOL ok = ((BOOL(*)(id, SEL, uint64_t))objc_msgSend)(dev, supports, count);
      NSArray *t = trappedAndReset();
      if (t.count) {
        positions[n.stringValue] = @{@"trapped": t};
        continue;
      }
      if (!ok) {
        positions[n.stringValue] = @{@"unsupported": @YES};
        continue;
      }
      MTLSamplePosition p[64];
      ((void (*)(id, SEL, MTLSamplePosition *, NSUInteger))objc_msgSend)(dev, getPositions, p, count);
      NSMutableArray *v = [NSMutableArray array];
      for (NSUInteger i = 0; i < count; i++) {
        [v addObject:@(p[i].x)];
        [v addObject:@(p[i].y)];
      }
      positions[n.stringValue] = result(v);
    }
    out[@"sample_positions"] = positions;

    NSOperatingSystemVersion os = [NSProcessInfo processInfo].operatingSystemVersion;
    out[@"host"] = @{
      @"os": [NSString stringWithFormat:@"%ld.%ld.%ld", (long)os.majorVersion, (long)os.minorVersion, (long)os.patchVersion],
      @"metal_device_instance_size": @(class_getInstanceSize(base)),
    };
    NSData *json = [NSJSONSerialization dataWithJSONObject:out options:NSJSONWritingSortedKeys error:nil];
    fwrite(json.bytes, 1, json.length, stdout);
    fflush(stdout);
    _exit(0);
  }
}
