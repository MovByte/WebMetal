//! A JSON reader for the reference outputs (`tests/reference/FORMAT.md`: UTF-8 JSON, floating-point values written as
//! hexadecimal bit patterns, so every number is an integer).

use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i128),
    Str(String),
    List(Vec<Json>),
    Map(BTreeMap<String, Json>),
}

impl Json {
    pub fn parse(text: &str) -> Json {
        let mut p = Parser { b: text.as_bytes(), i: 0 };
        let v = p.value();
        p.ws();
        assert_eq!(p.i, p.b.len(), "trailing bytes after the JSON value");
        v
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Map(m) => m.get(key),
            _ => None,
        }
    }

    pub fn str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn int(&self) -> Option<i128> {
        match self {
            Json::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn bool(&self) -> Option<bool> {
        match self {
            Json::Bool(v) => Some(*v),
            // Booleans of the harness's own records are written as 0 or 1 in some places.
            Json::Int(v) => Some(*v != 0),
            _ => None,
        }
    }

    pub fn list(&self) -> &[Json] {
        match self {
            Json::List(l) => l,
            _ => &[],
        }
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) {
        self.ws();
        assert_eq!(self.b[self.i], c, "JSON: expected {:?} at {}", c as char, self.i);
        self.i += 1;
    }

    fn value(&mut self) -> Json {
        self.ws();
        match self.b[self.i] {
            b'{' => {
                self.i += 1;
                let mut m = BTreeMap::new();
                self.ws();
                if self.b[self.i] == b'}' {
                    self.i += 1;
                    return Json::Map(m);
                }
                loop {
                    self.ws();
                    let Json::Str(k) = self.value() else { panic!("JSON: a key is not a string") };
                    self.eat(b':');
                    let v = self.value();
                    m.insert(k, v);
                    self.ws();
                    match self.b[self.i] {
                        b',' => self.i += 1,
                        b'}' => {
                            self.i += 1;
                            return Json::Map(m);
                        }
                        c => panic!("JSON: unexpected {:?} in an object", c as char),
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut l = Vec::new();
                self.ws();
                if self.b[self.i] == b']' {
                    self.i += 1;
                    return Json::List(l);
                }
                loop {
                    l.push(self.value());
                    self.ws();
                    match self.b[self.i] {
                        b',' => self.i += 1,
                        b']' => {
                            self.i += 1;
                            return Json::List(l);
                        }
                        c => panic!("JSON: unexpected {:?} in an array", c as char),
                    }
                }
            }
            b'"' => {
                self.i += 1;
                let mut s = String::new();
                loop {
                    let c = self.b[self.i];
                    self.i += 1;
                    match c {
                        b'"' => return Json::Str(s),
                        b'\\' => {
                            let e = self.b[self.i];
                            self.i += 1;
                            match e {
                                b'n' => s.push('\n'),
                                b't' => s.push('\t'),
                                b'r' => s.push('\r'),
                                b'u' => {
                                    let h = std::str::from_utf8(&self.b[self.i..self.i + 4]).unwrap();
                                    self.i += 4;
                                    s.push(char::from_u32(u32::from_str_radix(h, 16).unwrap()).unwrap_or('?'));
                                }
                                other => s.push(other as char),
                            }
                        }
                        _ => {
                            // Copy a UTF-8 sequence whole.
                            let start = self.i - 1;
                            let mut end = self.i;
                            while end < self.b.len() && (self.b[end] & 0xc0) == 0x80 {
                                end += 1;
                            }
                            s.push_str(std::str::from_utf8(&self.b[start..end]).unwrap());
                            self.i = end;
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
                while self.i < self.b.len() && (self.b[self.i] == b'-' || self.b[self.i].is_ascii_digit()) {
                    self.i += 1;
                }
                let text = std::str::from_utf8(&self.b[start..self.i]).unwrap();
                Json::Int(text.parse().unwrap_or_else(|_| panic!("JSON: {text:?} is not an integer")))
            }
        }
    }
}
