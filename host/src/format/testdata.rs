//! Readers for the reference data the format tests compare against: the Metal reference outputs
//! (`tests/reference/goldens`, format `webmetal-reference/1`, see `tests/reference/FORMAT.md`) and the PVRTC
//! measurement (`tables/measure/pvrtc_mac.toml`).

use std::collections::BTreeMap;
use std::path::PathBuf;

/// A JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(BTreeMap<String, Json>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(m) => m.get(key),
            _ => None,
        }
    }

    pub fn str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn num(&self) -> Option<f64> {
        match self {
            Json::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(a) => Some(a),
            _ => None,
        }
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn expect(&mut self, c: u8) {
        self.ws();
        assert_eq!(self.s.get(self.i), Some(&c), "JSON: expected {:?} at {}", c as char, self.i);
        self.i += 1;
    }

    fn value(&mut self) -> Json {
        self.ws();
        match self.s[self.i] {
            b'{' => {
                self.i += 1;
                let mut m = BTreeMap::new();
                self.ws();
                if self.s[self.i] == b'}' {
                    self.i += 1;
                    return Json::Object(m);
                }
                loop {
                    self.ws();
                    let Json::String(k) = self.value() else { panic!("JSON: object key") };
                    self.expect(b':');
                    let v = self.value();
                    m.insert(k, v);
                    self.ws();
                    match self.s[self.i] {
                        b',' => self.i += 1,
                        b'}' => {
                            self.i += 1;
                            return Json::Object(m);
                        }
                        c => panic!("JSON: unexpected {:?}", c as char),
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.s[self.i] == b']' {
                    self.i += 1;
                    return Json::Array(a);
                }
                loop {
                    a.push(self.value());
                    self.ws();
                    match self.s[self.i] {
                        b',' => self.i += 1,
                        b']' => {
                            self.i += 1;
                            return Json::Array(a);
                        }
                        c => panic!("JSON: unexpected {:?}", c as char),
                    }
                }
            }
            b'"' => {
                self.i += 1;
                let mut out = String::new();
                loop {
                    let c = self.s[self.i];
                    self.i += 1;
                    match c {
                        b'"' => return Json::String(out),
                        b'\\' => {
                            let e = self.s[self.i];
                            self.i += 1;
                            match e {
                                b'"' => out.push('"'),
                                b'\\' => out.push('\\'),
                                b'/' => out.push('/'),
                                b'n' => out.push('\n'),
                                b't' => out.push('\t'),
                                b'r' => out.push('\r'),
                                b'b' => out.push('\u{8}'),
                                b'f' => out.push('\u{c}'),
                                b'u' => {
                                    let h = std::str::from_utf8(&self.s[self.i..self.i + 4]).unwrap();
                                    self.i += 4;
                                    let mut cp = u32::from_str_radix(h, 16).unwrap();
                                    if (0xd800..0xdc00).contains(&cp) {
                                        assert_eq!(&self.s[self.i..self.i + 2], b"\\u");
                                        let l = std::str::from_utf8(&self.s[self.i + 2..self.i + 6]).unwrap();
                                        self.i += 6;
                                        cp = 0x10000
                                            + ((cp - 0xd800) << 10)
                                            + (u32::from_str_radix(l, 16).unwrap() - 0xdc00);
                                    }
                                    out.push(char::from_u32(cp).unwrap());
                                }
                                _ => panic!("JSON: escape {:?}", e as char),
                            }
                        }
                        _ => {
                            // Copy a whole UTF-8 sequence.
                            let start = self.i - 1;
                            let len = match c {
                                0..=0x7f => 1,
                                0xc0..=0xdf => 2,
                                0xe0..=0xef => 3,
                                _ => 4,
                            };
                            out.push_str(std::str::from_utf8(&self.s[start..start + len]).unwrap());
                            self.i = start + len;
                        }
                    }
                }
            }
            b't' => {
                self.i += 4;
                Json::Bool(true)
            }
            b'f' => {
                self.i += 5;
                Json::Bool(false)
            }
            b'n' => {
                self.i += 4;
                Json::Null
            }
            _ => {
                let start = self.i;
                while self.i < self.s.len() && matches!(self.s[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
                {
                    self.i += 1;
                }
                Json::Number(std::str::from_utf8(&self.s[start..self.i]).unwrap().parse().unwrap())
            }
        }
    }
}

pub fn parse_json(text: &str) -> Json {
    let mut p = Parser { s: text.as_bytes(), i: 0 };
    let v = p.value();
    p.ws();
    assert_eq!(p.i, p.s.len(), "JSON: trailing data");
    v
}

pub fn repo_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

/// The reference device key whose outputs the tests compare against.
pub const GOLDEN_DEVICE: &str = "apple-m4-macos-27.0.1-26a434";

/// A case file of the Metal reference outputs (release variant).
pub struct Case {
    pub json: Json,
}

impl Case {
    pub fn load(case: &str) -> Case {
        let path = repo_path(&format!("tests/reference/goldens/{GOLDEN_DEVICE}/release/{case}.json"));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        Case { json: parse_json(&text) }
    }

    pub fn status(&self) -> &str {
        self.json.get("status").and_then(Json::str).unwrap()
    }

    pub fn record(&self, name: &str) -> Option<&Json> {
        self.json.get("records")?.array()?.iter().find(|r| r.get("name").and_then(Json::str) == Some(name))
    }

    /// The bytes of a `data` record.
    pub fn bytes(&self, name: &str) -> Vec<u8> {
        let hex = self.record(name).and_then(|r| r.get("hex")).and_then(Json::str).unwrap_or_else(|| panic!("{name}"));
        hex_bytes(hex)
    }

    /// The comparison rule of a record (`exact`, `none`, ...).
    pub fn rule(&self, name: &str) -> &str {
        self.record(name).and_then(|r| r.get("rule")).and_then(|r| r.get("compare")).and_then(Json::str).unwrap()
    }

    /// The `f32` elements of a `data` record.
    pub fn f32s(&self, name: &str) -> Vec<f32> {
        self.bytes(name).chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    pub fn layout_u32(&self, name: &str, key: &str) -> u32 {
        self.record(name).and_then(|r| r.get("layout")).and_then(|l| l.get(key)).and_then(Json::num).unwrap() as u32
    }
}

pub fn hex_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect()
}

/// The decoded texels of one case of `tables/measure/pvrtc_mac.toml`.
pub fn pvrtc_measurement(id: &str) -> Vec<f32> {
    let path = repo_path("tables/measure/pvrtc_mac.toml");
    let text = std::fs::read_to_string(path).unwrap();
    let start = text.find(&format!("id = \"{id}\"")).unwrap_or_else(|| panic!("{id}"));
    let line = text[start..].lines().find(|l| l.starts_with("texels_rgba = [")).unwrap();
    let list = line.trim_start_matches("texels_rgba = [").trim_end_matches(']');
    list.split(", ").map(|v| v.parse::<f32>().unwrap()).collect()
}
