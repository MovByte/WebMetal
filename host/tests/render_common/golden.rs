//! The Metal reference outputs (`tests/reference/goldens`, release variant) and their comparison with what a replay
//! measured, under the rules of `tests/reference/FORMAT.md` ("Comparing two runs", step 5): every output record of the
//! reference must be measured, with the same kind, type and layout, and satisfy its rule; a replay may not measure
//! records the reference lacks.

use std::collections::BTreeMap;
use std::path::PathBuf;

use super::json::Json;
use crate::mem_common::GOLDEN_DEVICE;

pub struct Golden {
    pub case: String,
    pub file: Json,
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

impl Golden {
    pub fn load(case: &str) -> Golden {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/reference/goldens")
            .join(GOLDEN_DEVICE)
            .join("release")
            .join(format!("{case}.json"));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        Golden { case: case.to_string(), file: Json::parse(&text) }
    }

    pub fn status(&self) -> &str {
        self.file.get("status").and_then(Json::str).unwrap_or("")
    }

    pub fn records(&self) -> &[Json] {
        self.file.get("records").map(Json::list).unwrap_or(&[])
    }

    pub fn record(&self, name: &str) -> Option<&Json> {
        self.records().iter().find(|r| r.get("name").and_then(Json::str) == Some(name))
    }

    /// The bytes of a `data` record.
    pub fn data(&self, name: &str) -> Vec<u8> {
        hex(self
            .record(name)
            .and_then(|r| r.get("hex"))
            .and_then(Json::str)
            .unwrap_or_else(|| panic!("{}: no data record {name}", self.case)))
    }

    /// The requirement records, in order.
    pub fn requirements(&self) -> Vec<(String, bool)> {
        self.records()
            .iter()
            .filter(|r| r.get("kind").and_then(Json::str) == Some("require"))
            .map(|r| {
                (r.get("name").and_then(Json::str).unwrap().to_string(), r.get("value").and_then(Json::bool).unwrap())
            })
            .collect()
    }
}

/// A measured output.
#[derive(Clone, Debug)]
pub enum Measured {
    Data { elem: &'static str, bytes: Vec<u8>, layout: Option<Json> },
    U64(u64),
    Bool(bool),
}

/// What a replay of a case measured.
#[derive(Default)]
pub struct Candidate {
    pub records: Vec<(String, Measured)>,
}

/// The counts of a comparison, by rule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    pub exact: usize,
    pub none: usize,
    pub device: usize,
    pub bounded: usize,
    pub property: usize,
}

impl Outcome {
    pub fn total(&self) -> usize {
        self.exact + self.none + self.device + self.bounded + self.property
    }
}

impl std::fmt::Display for Outcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} records: {} exact, {} none, {} device, {} bounded, {} property",
            self.total(),
            self.exact,
            self.none,
            self.device,
            self.bounded,
            self.property
        )
    }
}

/// A `layout` object of a texture data record (FORMAT.md, "Layouts").
pub fn layout(format: &str, fields: &[(&str, u64)]) -> Json {
    let mut m = BTreeMap::new();
    m.insert("format".to_string(), Json::Str(format.to_string()));
    for (k, v) in fields {
        m.insert(k.to_string(), Json::Int(*v as i128));
    }
    Json::Map(m)
}

impl Candidate {
    pub fn data(&mut self, name: &str, elem: &'static str, bytes: &[u8]) {
        self.records.push((name.to_string(), Measured::Data { elem, bytes: bytes.to_vec(), layout: None }));
    }

    pub fn data_layout(&mut self, name: &str, elem: &'static str, bytes: &[u8], layout: Json) {
        self.records.push((name.to_string(), Measured::Data { elem, bytes: bytes.to_vec(), layout: Some(layout) }));
    }

    pub fn u64(&mut self, name: &str, v: u64) {
        self.records.push((name.to_string(), Measured::U64(v)));
    }

    pub fn bool(&mut self, name: &str, v: bool) {
        self.records.push((name.to_string(), Measured::Bool(v)));
    }

    /// Compare with the reference. Returns the counts, or every disagreement.
    pub fn compare(&self, golden: &Golden) -> Result<Outcome, String> {
        let mut out = Outcome::default();
        let mut problems = Vec::new();
        let mut seen = Vec::new();
        for r in golden.records() {
            let kind = r.get("kind").and_then(Json::str).unwrap_or("");
            if !matches!(kind, "value" | "data" | "error" | "call") {
                continue;
            }
            let name = r.get("name").and_then(Json::str).unwrap();
            seen.push(name.to_string());
            let rule = r.get("rule").and_then(|r| r.get("compare")).and_then(Json::str).unwrap_or("");
            let Some((_, m)) = self.records.iter().find(|(n, _)| n == name) else {
                problems.push(format!("{name}: not measured"));
                continue;
            };
            // Kind and type.
            let ty = r.get("type").and_then(Json::str);
            match (kind, m) {
                ("data", Measured::Data { elem, bytes, layout }) => {
                    if r.get("elem").and_then(Json::str) != Some(*elem) {
                        problems.push(format!("{name}: element type {elem}, reference {:?}", r.get("elem")));
                        continue;
                    }
                    if r.get("layout").is_some() && r.get("layout") != layout.as_ref() {
                        problems.push(format!("{name}: layout {layout:?}, reference {:?}", r.get("layout")));
                        continue;
                    }
                    match rule {
                        "exact" => {
                            let want = hex(r.get("hex").and_then(Json::str).unwrap());
                            if &want != bytes {
                                problems.push(format!(
                                    "{name}: measured {} differs from the reference {}",
                                    hexstr(bytes),
                                    hexstr(&want)
                                ));
                                continue;
                            }
                            out.exact += 1;
                        }
                        "none" => out.none += 1,
                        "device" => out.device += 1,
                        "bounded" => match bounded(r, elem, bytes) {
                            Ok(()) => out.bounded += 1,
                            Err(e) => problems.push(format!("{name}: {e}")),
                        },
                        other => problems.push(format!("{name}: rule {other} on a data record")),
                    }
                }
                ("value", Measured::U64(v)) if ty == Some("u64") => {
                    let want = r.get("value").and_then(Json::int).unwrap();
                    match rule {
                        "exact" if want == *v as i128 => out.exact += 1,
                        "exact" => problems.push(format!("{name}: measured {v}, reference {want}")),
                        "none" => out.none += 1,
                        "device" => out.device += 1,
                        "property" => {
                            let values = r.get("rule").and_then(|r| r.get("values")).map(Json::list).unwrap_or(&[]);
                            if values.iter().any(|x| x.int() == Some(*v as i128)) {
                                out.property += 1;
                            } else {
                                problems.push(format!("{name}: {v} is not one of {values:?}"));
                            }
                        }
                        other => problems.push(format!("{name}: rule {other}")),
                    }
                }
                ("value", Measured::Bool(v)) if ty == Some("bool") => {
                    let want = r.get("value").and_then(Json::bool).unwrap();
                    match rule {
                        "exact" if want == *v => out.exact += 1,
                        "exact" => problems.push(format!("{name}: measured {v}, reference {want}")),
                        "none" => out.none += 1,
                        "device" => out.device += 1,
                        other => problems.push(format!("{name}: rule {other}")),
                    }
                }
                (k, m) => problems.push(format!("{name}: measured {m:?} for a {k} record of type {ty:?}")),
            }
        }
        for (n, _) in &self.records {
            if !seen.contains(n) {
                problems.push(format!("{n}: measured, but the reference has no such record"));
            }
        }
        if problems.is_empty() { Ok(out) } else { Err(problems.join("\n")) }
    }
}

fn hexstr(b: &[u8]) -> String {
    let s: String = b.iter().map(|x| format!("{x:02x}")).collect();
    if s.len() > 160 { format!("{}...({} bytes)", &s[..160], b.len()) } else { s }
}

/// No render case of the reference uses bounded rules; the replays here do not apply them.
fn bounded(_: &Json, elem: &str, _: &[u8]) -> Result<(), String> {
    Err(format!("bounded comparison ({elem}) is not used by the render cases and not implemented here"))
}
