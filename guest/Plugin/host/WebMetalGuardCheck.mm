// Checks of the base-class guard (WebMetalGuard.mm) against 21H461's Metal.framework base classes:
//
//     WebMetalGuardCheck
//
// The check defines every base class of guest/Generated/WebMetalTargetSurface.h at run time from 21H461's facts as the
// tables record them: its superclass, its instance variables at their offsets, ending where its class_ro_t
// instanceSize says, and every method Metal.framework implements for it, with its encoding. The classes take the name
// of the base after "WebMetalGuardCheck." (this Mac's Metal.framework has the names), and the guard is given the
// tables under those names. It must accept these classes, though class_getInstanceSize answers a word-aligned size the
// tables do not record (_MTLCommandQueue: 345, answered 352) and NSObject answers selectors the tables record as absent
// (-[_MTLResource description]). Then it must refuse a class changed in each way a layout can change: its size, an
// instance variable inside its last word, its superclass, a method the plugin overrides that is missing, a method's
// encoding, and an instance or class method of its own that 21H461's class does not have. A method a superclass of
// the same image implements must count, and one only NSObject answers must not. Prints one line per failure and PASS
// or FAIL.

#import <Foundation/Foundation.h>
#import <objc/runtime.h>

#include "../../Generated/WebMetalTargetSurface.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <set>
#include <string>
#include <vector>

using namespace webmetal;

namespace {

constexpr size_t kBaseCount = sizeof target::kBases / sizeof *target::kBases;

unsigned gFailures;
std::deque<std::string> gNames;  // the names the classes and the tables take
std::vector<BaseLayout> gBases;  // target::kBases under those names, in the same order

void failure(const std::string &text)
{
    gFailures++;
    printf("FAIL %s\n", text.c_str());
}

void method(id, SEL) {}

const char *keep(const std::string &name)
{
    gNames.push_back(name);
    return gNames.back().c_str();
}

const char *checkName(const char *name)
{
    for (const BaseLayout &base : target::kBases)
        if (!strcmp(base.name, name))
            return keep(std::string("WebMetalGuardCheck.") + name);
    return name;
}

// 21H461's base `name` (its target::kBases name) as the check defines it, or nullptr.
const BaseLayout *findBase(const char *name)
{
    for (size_t i = 0; i < kBaseCount; i++)
        if (!strcmp(target::kBases[i].name, name))
            return &gBases[i];
    return nullptr;
}

const BaseLayout &requireBase(const char *name)
{
    const BaseLayout *base = findBase(name);
    if (!base) {
        fprintf(stderr, "the generated tables have no base %s\n", name);
        exit(2);
    }
    return *base;
}

const BaseLayout &original(const BaseLayout &base)
{
    return target::kBases[&base - gBases.data()];
}

// How a class defined from a base's facts differs from them.
struct Change {
    const char *superclass = nullptr;  // another superclass
    size_t addedIvarSize = 0;          // an instance variable of this many bytes after the last
    const char *addedIvarType = nullptr;
    const BaseSelector *dropped = nullptr;  // a method left out
    const BaseSelector *added = nullptr;    // a method added, with the encoding it carries
    const char *addedTypes = nullptr;
};

// Where the instance variables of a class derived from `superclass` start: where the superclass's end.
bool superclassEnd(const char *superclass, size_t *end)
{
    if (!strcmp(superclass, "NSObject")) {
        *end = class_getInstanceSize([NSObject class]);  // isa alone: the size is not rounded
        return true;
    }
    for (const BaseLayout &base : gBases)
        if (!strcmp(base.name, superclass)) {
            *end = base.instanceSize;
            return true;
        }
    return false;
}

// Defines class `name` as `base`'s facts describe it, with `change`. Returns Nil after a failure line when the runtime
// cannot reproduce the facts.
Class build(const BaseLayout &base, const char *name, const Change &change = {})
{
    const char *superclassName = change.superclass ? change.superclass : base.superclass;
    Class superclass = objc_getClass(superclassName);
    size_t end = 0;
    if (!superclass || !superclassEnd(base.superclass, &end)) {
        failure(std::string(name) + ": no superclass " + superclassName + " to define it on");
        return Nil;
    }
    Class cls = objc_allocateClassPair(superclass, name, 0);
    if (!cls) {
        failure(std::string(name) + ": the class cannot be allocated");
        return Nil;
    }
    // class_addIvar places a variable at the class's unaligned size rounded up to the given alignment and ends the
    // class after its size: each variable gets the smallest alignment that places it at 21H461's offset, and the bytes
    // up to the next one (to the instanceSize for the last).
    for (size_t i = 0; i < base.ivarCount; i++) {
        const BaseIvar &ivar = base.ivars[i];
        const size_t next = i + 1 < base.ivarCount ? size_t(base.ivars[i + 1].offset) : base.instanceSize;
        uint8_t alignment = 0;
        while (alignment < 16 && ((end + (size_t(1) << alignment) - 1) & ~((size_t(1) << alignment) - 1)) !=
                                     size_t(ivar.offset))
            alignment++;
        if (alignment == 16 || next <= size_t(ivar.offset) ||
            !class_addIvar(cls, ivar.name, next - ivar.offset, alignment, ivar.type)) {
            failure(std::string(name) + ": instance variable " + ivar.name + " cannot be placed at " +
                    std::to_string(ivar.offset));
            objc_disposeClassPair(cls);
            return Nil;
        }
        end = next;
    }
    if (end != base.instanceSize) {
        failure(std::string(name) + ": its instance variables end at " + std::to_string(end) + ", not at " +
                std::to_string(base.instanceSize));
        objc_disposeClassPair(cls);
        return Nil;
    }
    if (change.addedIvarSize && !class_addIvar(cls, "_added", change.addedIvarSize, 0, change.addedIvarType)) {
        failure(std::string(name) + ": the added instance variable cannot be placed");
        objc_disposeClassPair(cls);
        return Nil;
    }
    auto add = [&](const BaseSelector &s, const char *types) {
        Class owner = s.kind == SelectorKind::Class ? object_getClass(cls) : cls;
        if (!class_addMethod(owner, sel_registerName(s.name), (IMP)method, types))
            failure(std::string(name) + ": " + s.name + " cannot be added");
    };
    for (size_t i = 0; i < base.selectorCount; i++) {
        const BaseSelector &s = base.selectors[i];
        if (s.name && s.types && &s != change.dropped && &s != change.added)
            add(s, s.types);
    }
    if (change.added)
        add(*change.added, change.addedTypes);
    objc_registerClassPair(cls);
    return cls;
}

// Defines a base and, first, the bases it derives from.
bool define(const BaseLayout &base, std::set<std::string> *defined)
{
    if (defined->count(base.name))
        return true;
    for (const BaseLayout &other : gBases)
        if (!strcmp(other.name, base.superclass) && !define(other, defined))
            return false;
    if (!build(base, base.name))
        return false;
    defined->insert(base.name);
    return true;
}

// The guard's answer for one base layout under another name.
bool check(const BaseLayout &base, const char *name, std::string *difference)
{
    BaseLayout renamed = base;
    renamed.name = name;
    const LayoutSet set = {"changed", &renamed, 1};
    return CheckLayouts(set, difference);
}

// A class defined from `base`'s facts with `change` must be refused, with a difference that contains `expected`.
void refused(const char *what, const BaseLayout &base, const Change &change, const std::string &expected)
{
    static unsigned serial;
    const char *name = keep(std::string(base.name) + "." + std::to_string(++serial));
    if (!build(base, name, change))
        return;
    std::string difference;
    if (check(base, name, &difference))
        failure(std::string("the guard accepts ") + what);
    else if (difference.find(expected) == std::string::npos)
        failure(std::string("the guard refuses ") + what + " for another difference: " + difference);
    else
        printf("refused %s: %s\n", what, difference.c_str());
}

const BaseSelector *selector(const BaseLayout &base, SelectorKind kind, const char *name)
{
    for (size_t i = 0; i < base.selectorCount; i++)
        if (base.selectors[i].name && base.selectors[i].kind == kind && !strcmp(base.selectors[i].name, name))
            return &base.selectors[i];
    return nullptr;
}

// A selector Metal.framework implements for `base` that a plugin class overrides (a driver selector of a role over
// the base), which NSObject answers too when `nsobject` is set and does not otherwise.
const BaseSelector *overridden(const BaseLayout &base, bool nsobject)
{
    for (size_t i = 0; i < base.selectorCount; i++) {
        const BaseSelector &s = base.selectors[i];
        if (!s.name || !s.types)
            continue;
        SEL sel = sel_registerName(s.name);
        const bool answered = s.kind == SelectorKind::Class ? class_getClassMethod([NSObject class], sel) != nullptr
                                                            : class_getInstanceMethod([NSObject class], sel) != nullptr;
        if (answered != nsobject)
            continue;
        for (const Role &role : target::kRoles) {
            if (strcmp(role.base, original(base).name))
                continue;
            for (size_t k = 0; k < role.count; k++)
                if (role.selectors[k].kind == s.kind && !strcmp(role.selectors[k].name, s.name))
                    return &s;
        }
    }
    return nullptr;
}

void checkTarget()
{
    std::set<std::string> defined;
    for (const BaseLayout &base : gBases)
        if (!define(base, &defined))
            return;
    const BaseLayout &queue = requireBase("_MTLCommandQueue");
    const size_t answered = class_getInstanceSize(objc_getClass(queue.name));
    if (answered == queue.instanceSize)
        failure("_MTLCommandQueue's class_getInstanceSize equals its class_ro_t instanceSize: the check no longer "
                "covers the rounding");
    const BaseLayout &resource = requireBase("_MTLResource");
    const BaseSelector *description = selector(resource, SelectorKind::Instance, "description");
    if (!description || description->types ||
        !class_getInstanceMethod(objc_getClass(resource.name), sel_registerName("description"))) {
        failure("-[_MTLResource description] is not a selector the tables record as absent and NSObject answers: the "
                "check no longer covers inherited methods");
        return;
    }
    std::string difference;
    const LayoutSet set = {"iPadOS 17.7.11 (21H461)", gBases.data(), gBases.size()};
    if (!CheckLayouts(set, &difference))
        failure("the guard refuses 21H461's own layouts: " + difference);
    else
        printf("accepted: the %zu base classes as 21H461 defines them (_MTLCommandQueue: instanceSize %zu, "
               "class_getInstanceSize %zu; -[_MTLResource description] absent though NSObject answers it)\n",
               set.count, queue.instanceSize, answered);

    Change larger;
    larger.addedIvarSize = 8;
    larger.addedIvarType = "Q";
    refused("an instance size changed by a word", queue, larger, "instance size");
    Change tail;
    tail.addedIvarSize = 1;
    tail.addedIvarType = "B";
    refused("an instance variable added inside the last word (the same class_getInstanceSize)", queue, tail,
            "instance variables");
    Change superclass;
    superclass.superclass = "NSObject";
    refused("another superclass", resource, superclass, "superclass");

    const BaseLayout &device = requireBase("_MTLDevice");
    if (const BaseSelector *s = overridden(device, false)) {
        Change missing;
        missing.dropped = s;
        refused("a missing method the plugin overrides", device, missing, std::string(s->name) + " is missing");
        Change encoding;
        encoding.added = s;
        encoding.addedTypes = strcmp(s->types, "v16@0:8") ? "v16@0:8" : "@16@0:8";
        refused("a method with another encoding", device, encoding, std::string(s->name) + " has type encoding");
    } else {
        failure("_MTLDevice has no selector the plugin overrides that Metal.framework implements");
    }
    Change own;
    own.added = description;
    own.addedTypes = "@16@0:8";
    refused("an instance method of its own 21H461's class does not have", resource, own,
            "-description is implemented, expected no implementation");
    const BaseSelector *classAbsent = nullptr;
    for (size_t i = 0; !classAbsent && i < resource.selectorCount; i++)
        if (resource.selectors[i].kind == SelectorKind::Class && !resource.selectors[i].types)
            classAbsent = &resource.selectors[i];
    if (classAbsent) {
        Change ownClass;
        ownClass.added = classAbsent;
        ownClass.addedTypes = "@16@0:8";
        refused("a class method of its own 21H461's class does not have", resource, ownClass,
                std::string("+") + classAbsent->name + " is implemented");
    } else {
        failure("_MTLResource has no class selector the tables record as absent");
    }
}

// A method of a superclass from the same image counts as the framework's; the same method expected absent is refused,
// and a method only the root class answers is missing.
void checkSuperclassMethods()
{
    Class above = objc_allocateClassPair([NSObject class], "WebMetalGuardCheck.Above", 0);
    class_addMethod(above, sel_registerName("label"), (IMP)method, "@16@0:8");
    objc_registerClassPair(above);
    Class below = objc_allocateClassPair(above, "WebMetalGuardCheck.Below", 0);
    objc_registerClassPair(below);
    const BaseSelector present[] = {{SelectorKind::Instance, "label", "@16@0:8"}};
    const BaseSelector absent[] = {{SelectorKind::Instance, "label", nullptr}};
    BaseLayout layout = {"WebMetalGuardCheck.Below", "WebMetalGuardCheck.Above", 8, nullptr, 0, present, 1};
    std::string difference;
    if (!check(layout, layout.name, &difference))
        failure("the guard refuses a method its superclass in the same image implements: " + difference);
    else
        printf("accepted: a method of a superclass in the same image\n");
    layout.selectors = absent;
    if (check(layout, layout.name, &difference))
        failure("the guard accepts a method its superclass in the same image implements as absent");
    else
        printf("refused a superclass's method in the same image as absent: %s\n", difference.c_str());
    const BaseSelector root[] = {{SelectorKind::Instance, "description", "@16@0:8"}};
    layout.selectors = root;
    if (check(layout, layout.name, &difference))
        failure("the guard accepts NSObject's -description as the framework's");
    else
        printf("refused a method only NSObject answers: %s\n", difference.c_str());
}

}  // namespace

int main()
{
    @autoreleasepool {
        for (const BaseLayout &base : target::kBases) {
            BaseLayout renamed = base;
            renamed.name = checkName(base.name);
            renamed.superclass = checkName(base.superclass);
            gBases.push_back(renamed);
        }
        checkTarget();
        checkSuperclassMethods();
        if (gFailures) {
            printf("FAIL: %u failures\n", gFailures);
            return 1;
        }
        printf("PASS: the guard accepts 21H461's base classes and refuses each change of them\n");
    }
    return 0;
}
