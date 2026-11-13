// The base-class guard: the plugin's classes subclass Metal.framework classes that no SDK header describes, so before
// a device is created the running classes are compared with the layouts the plugin was built against (WebMetalSurface.h).
// Any difference refuses the device; nothing is adapted to a framework the plugin was not built for.

#import "WebMetalSurface.h"

#import <Foundation/Foundation.h>
#import <objc/runtime.h>

#include <cstdlib>
#include <cstring>

namespace webmetal {

namespace {

std::string describe(const BaseLayout &base, const std::string &what)
{
    return std::string(base.name) + ": " + what;
}

bool checkIvars(Class cls, const BaseLayout &base, std::string *difference)
{
    unsigned count = 0;
    Ivar *ivars = class_copyIvarList(cls, &count);
    bool same = count == base.ivarCount;
    if (!same)
        *difference = describe(base, std::to_string(count) + " instance variables, expected " +
                                         std::to_string(base.ivarCount));
    for (unsigned i = 0; same && i < count; i++) {
        const BaseIvar &want = base.ivars[i];
        const char *name = ivar_getName(ivars[i]);
        const char *type = ivar_getTypeEncoding(ivars[i]);
        const ptrdiff_t offset = ivar_getOffset(ivars[i]);
        if (!name || strcmp(name, want.name) != 0 || offset != want.offset || !type || strcmp(type, want.type) != 0) {
            *difference = describe(base, "instance variable " + std::to_string(i) + " is " + (name ? name : "?") +
                                             " at " + std::to_string(offset) + ", expected " + want.name + " at " +
                                             std::to_string(want.offset) + " with type " + want.type);
            same = false;
        }
    }
    free(ivars);
    return same;
}

bool sameImage(const char *a, const char *b)
{
    return a == b || (a && b && strcmp(a, b) == 0);
}

bool checkSelectors(Class cls, const BaseLayout &base, std::string *difference)
{
    for (size_t i = 0; i < base.selectorCount; i++) {
        const BaseSelector &want = base.selectors[i];
        if (!want.name)
            continue;
        Method method = FrameworkMethod(cls, want.kind, want.name);
        const char *sign = want.kind == SelectorKind::Class ? "+" : "-";
        if (!method != !want.types) {
            *difference = describe(base, std::string(sign) + want.name + (method ? " is implemented" : " is missing") +
                                             ", expected " + (want.types ? "it" : "no implementation"));
            return false;
        }
        const char *types = method ? method_getTypeEncoding(method) : nullptr;
        if (method && (!types || strcmp(types, want.types) != 0)) {
            *difference = describe(base, std::string(sign) + want.name + " has type encoding " + (types ? types : "?") +
                                             ", expected " + want.types);
            return false;
        }
    }
    return true;
}

}  // namespace

Method FrameworkMethod(Class cls, SelectorKind kind, const char *name)
{
    const char *image = class_getImageName(cls);
    SEL selector = sel_registerName(name);
    for (Class link = cls; link && sameImage(class_getImageName(link), image); link = class_getSuperclass(link)) {
        unsigned count = 0;
        Method *methods = class_copyMethodList(kind == SelectorKind::Class ? object_getClass(link) : link, &count);
        Method found = nullptr;
        for (unsigned i = 0; !found && i < count; i++)
            if (method_getName(methods[i]) == selector)
                found = methods[i];
        free(methods);
        if (found)
            return found;
    }
    return nullptr;
}

bool CheckLayouts(const LayoutSet &layouts, std::string *difference)
{
    for (size_t i = 0; i < layouts.count; i++) {
        const BaseLayout &base = layouts.bases[i];
        Class cls = objc_getClass(base.name);
        if (!cls) {
            *difference = describe(base, "no such class");
            return false;
        }
        Class superclass = class_getSuperclass(cls);
        if (!superclass || strcmp(class_getName(superclass), base.superclass) != 0) {
            *difference = describe(base, std::string("superclass ") + (superclass ? class_getName(superclass) : "none") +
                                             ", expected " + base.superclass);
            return false;
        }
        const size_t size = class_getInstanceSize(cls);
        if (size != AlignedInstanceSize(base.instanceSize)) {
            *difference = describe(base, "instance size " + std::to_string(size) + ", expected " +
                                             std::to_string(AlignedInstanceSize(base.instanceSize)) + " (" +
                                             std::to_string(base.instanceSize) + " rounded up to a word)");
            return false;
        }
        if (!checkIvars(cls, base, difference) || !checkSelectors(cls, base, difference))
            return false;
    }
    return true;
}

}  // namespace webmetal
