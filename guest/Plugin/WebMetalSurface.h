// The shapes of the target-surface tables in guest/Generated/WebMetalTargetSurface.h, and the functions that apply
// them to the running process.

#pragma once

#include <objc/runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace webmetal {

enum class SelectorKind { Instance, Class };

// A selector the 21H461 driver class implements in its own layers (AGXMetalA10 or IOGPU.framework). The plugin class
// standing in for it implements it or answers it with a LOUD stub.
struct DriverSelector {
    SelectorKind kind;
    const char *name;
    const char *types;   // the 21H461 method's type encoding
    const char *layer;   // "agx" or "iogpu"
    const char *detail;  // why it is not implemented, when that is known; "" otherwise
};

// A driver class of AGXMetalA10 and the plugin class that stands in for it.
struct Role {
    const char *pluginClass;
    const char *targetClass;
    const char *base;  // the Metal.framework class both subclass
    const DriverSelector *selectors;
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

// The driver classes the plugin stands in for (guest/Plugin/roles.toml), from the generated tables.
const Role *Roles(size_t *count);

// Gives every plugin class a LOUD stub for each driver selector it does not implement itself. Returns false and
// describes the problem when a plugin class is missing or a stub cannot be added.
bool InstallLoudStubs(std::string *problem);

}  // namespace webmetal
