// The shapes of the target-surface tables in guest/Generated/WebMetalTargetSurface.h, and the functions that apply
// them to the running process.

#pragma once

#include <objc/runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace webmetal {

enum class SelectorKind { Instance, Class };

// A selector a class of the plugin carries for the 21H461 driver: one the AGXMetalA10 class of the same name defines,
// or, on the topmost AGXMetalA10 class of a chain, one IOGPU.framework defines above it. The plugin's class implements
// it or answers it with a LOUD stub.
struct DriverSelector {
    SelectorKind kind;
    const char *name;
    const char *types;    // the 21H461 method's type encoding
    const char *definer;  // the 21H461 class that defines it: the AGXMetalA10 class, or an IOGPU.framework class
    const char *detail;   // why it is not implemented, when that is known; "" otherwise
};

// An AGXMetalA10 class the plugin defines under the driver's name (guest/Plugin/roles.toml).
struct Role {
    const char *name;
    // The next AGXMetalA10 class of the 21H461 chain, which the plugin defines too, or `base` where the driver's chain
    // leaves the bundle (through IOGPU.framework's classes, which the plugin does not define).
    const char *superclass;
    const char *base;                 // the Metal.framework class (or NSObject) the chain reaches
    const DriverSelector *selectors;  // the driver selectors the class carries itself
    size_t count;
};

struct BaseIvar {
    const char *name;
    ptrdiff_t offset;
    const char *type;
};

// Whether Metal.framework implements a selector for the base class, and with which encoding (nullptr: it does not):
// the base class or one of its Metal.framework superclasses has the method in its own method lists. What the root
// class gives every class (NSObject's -description) is not the framework's (FrameworkMethod).
struct BaseSelector {
    SelectorKind kind;
    const char *name;
    const char *types;
};

// A Metal.framework base class as the plugin was built against it.
struct BaseLayout {
    const char *name;
    const char *superclass;
    size_t instanceSize;  // class_ro_t's instanceSize, where the instance variables end (objc4 unalignedInstanceSize)
    const BaseIvar *ivars;
    size_t ivarCount;
    const BaseSelector *selectors;  // the selectors of every role over this base: driver, inherited and absent ones
    size_t selectorCount;
};

struct LayoutSet {
    const char *source;  // what the layouts describe, for reports
    const BaseLayout *bases;
    size_t count;
};

// The layouts this build of the plugin requires (21H461's for the iOS bundle).
extern const LayoutSet kExpectedLayouts;

// What class_getInstanceSize answers for a class whose class_ro_t instanceSize is `unaligned`: objc4's
// alignedInstanceSize, word_align(unalignedInstanceSize()), the size rounded up to a multiple of 8 in 32 bits (21H461's
// libobjc, class_getInstanceSize at 0x1800fd99c: add w8, w8, #0x7; and w0, w8, #0xfffffff8).
constexpr size_t AlignedInstanceSize(size_t unaligned)
{
    return (static_cast<uint32_t>(unaligned) + 7u) & ~7u;
}

// The method Metal.framework gives `cls` for a selector, as the target audit reads the cache (UniXNU
// scripts/audit-target-web-metal-surface.py, base_layout): the first method of that name in the own method lists
// (class_copyMethodList, categories ahead of the class's methods) of `cls` and of each superclass its image defines,
// nearest first; for a class method, in their metaclasses' lists. The root class lies in another image, so what it
// gives every class (NSObject's -description) does not count. nullptr when there is none.
Method FrameworkMethod(Class cls, SelectorKind kind, const char *name);

// Compares the running Metal.framework with a layout set. Returns false and describes the first difference.
bool CheckLayouts(const LayoutSet &layouts, std::string *difference);

// The driver classes the plugin defines (guest/Plugin/roles.toml), from the generated tables.
const Role *Roles(size_t *count);

// The class named `name` that the plugin's own image defines, or Nil. AGXMetalA10 shares class names with other
// AGX drivers (AGXBuffer, AGXTexture, AGXTextureLayout, AGXPrincipalDevice), and objc_getClass answers whichever class
// of a name the runtime registered first, which may be another image's: this Mac's own driver beside the macOS test
// shim.
Class PluginClass(const char *name);

// Gives every class of the plugin a LOUD stub for each driver selector it carries and does not implement itself.
// Returns false and describes the problem when a class is missing or not over its superclass, or a stub cannot be
// added.
bool InstallLoudStubs(std::string *problem);

}  // namespace webmetal
