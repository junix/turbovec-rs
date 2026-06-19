//! Metadata filter compilation (filterql -> SQLite WHERE clause).

use anyhow::{anyhow, bail, Result};
use filterql::validate::Policy;
use filterql::{CmpOp, Compile, Value as FilterValue};
use rusqlite::types::Value as SqlValue;

#[derive(Debug, Clone)]
pub(crate) struct SqliteWhere {
    pub(crate) clause: String,
    pub(crate) params: Vec<SqlValue>,
}

#[derive(Debug, Default)]
pub(crate) struct SqliteFilterCompiler;

pub(crate) fn filter_policy() -> Policy {
    Policy::new()
        .max_depth(8)
        .max_comparisons(32)
        .max_in_list(256)
}

pub(crate) fn compile_filter(filter: &str) -> Result<SqliteWhere> {
    let expr = filterql::sql::parse(filter).context_filter("parsing metadata filter")?;
    filter_policy()
        .validate(&expr)
        .map_err(|report| anyhow!("invalid metadata filter:\n{}", report.render()))?;
    filterql::compile(&expr, &mut SqliteFilterCompiler)
        .map_err(|err| anyhow!("compiling metadata filter: {err}"))
}

pub(crate) fn validate_meta_field_name(field: &str) -> Result<()> {
    if field.is_empty() {
        bail!("metadata field name cannot be empty");
    }
    for part in field.split('.') {
        let mut chars = part.chars();
        let Some(first) = chars.next() else {
            bail!("metadata field `{field}` contains an empty path segment");
        };
        if !(first.is_ascii_alphabetic() || first == '_') {
            bail!("metadata field `{field}` must start each path segment with ASCII letter or `_`");
        }
        if chars.any(|c| !(c.is_ascii_alphanumeric() || c == '_')) {
            bail!("metadata field `{field}` contains unsupported characters");
        }
    }
    Ok(())
}

pub(crate) fn json_path_param(field: &str) -> Result<SqlValue> {
    validate_meta_field_name(field)?;
    Ok(SqlValue::Text(format!("$.{field}")))
}

pub(crate) fn sqlite_value(value: &FilterValue) -> Result<SqlValue> {
    Ok(match value {
        FilterValue::Str(s) => SqlValue::Text(s.clone()),
        FilterValue::Int(n) | FilterValue::Date(n) => SqlValue::Integer(*n),
        FilterValue::Float(n) => SqlValue::Real(*n),
        FilterValue::Bool(b) => SqlValue::Integer(i64::from(*b)),
        FilterValue::Null => SqlValue::Null,
        FilterValue::List(_) => bail!("list value cannot be bound as a scalar"),
    })
}

fn merge_params(mut parts: Vec<SqliteWhere>, separator: &str) -> SqliteWhere {
    let mut params = Vec::new();
    let clauses = parts
        .drain(..)
        .filter(|part| !part.clause.is_empty())
        .map(|part| {
            params.extend(part.params);
            part.clause
        })
        .collect::<Vec<_>>();

    SqliteWhere {
        clause: if clauses.is_empty() {
            String::new()
        } else {
            format!("({})", clauses.join(separator))
        },
        params,
    }
}

impl Compile for SqliteFilterCompiler {
    type Output = SqliteWhere;
    type Error = anyhow::Error;

    fn and(&mut self, parts: Vec<SqliteWhere>) -> Result<SqliteWhere> {
        Ok(merge_params(parts, " AND "))
    }

    fn or(&mut self, parts: Vec<SqliteWhere>) -> Result<SqliteWhere> {
        if parts.is_empty() {
            return Ok(SqliteWhere {
                clause: "(0 = 1)".to_string(),
                params: Vec::new(),
            });
        }
        Ok(merge_params(parts, " OR "))
    }

    fn not(&mut self, part: SqliteWhere) -> Result<SqliteWhere> {
        Ok(SqliteWhere {
            clause: format!("NOT ({})", part.clause),
            params: part.params,
        })
    }

    fn compare(&mut self, field: &str, op: CmpOp, value: &FilterValue) -> Result<SqliteWhere> {
        let path = json_path_param(field)?;
        match op {
            CmpOp::Exists => {
                let present = !matches!(value, FilterValue::Bool(false));
                Ok(SqliteWhere {
                    clause: format!(
                        "json_type(meta, ?) IS {}NULL",
                        if present { "NOT " } else { "" }
                    ),
                    params: vec![path],
                })
            }
            CmpOp::Eq if matches!(value, FilterValue::Null) => Ok(SqliteWhere {
                clause: "json_type(meta, ?) IS NULL".to_string(),
                params: vec![path],
            }),
            CmpOp::Ne if matches!(value, FilterValue::Null) => Ok(SqliteWhere {
                clause: "json_type(meta, ?) IS NOT NULL".to_string(),
                params: vec![path],
            }),
            CmpOp::Eq
            | CmpOp::Ne
            | CmpOp::Lt
            | CmpOp::Le
            | CmpOp::Gt
            | CmpOp::Ge
            | CmpOp::Like
            | CmpOp::NotLike => {
                let sql_op = match op {
                    CmpOp::Eq => "=",
                    CmpOp::Ne => "!=",
                    CmpOp::Lt => "<",
                    CmpOp::Le => "<=",
                    CmpOp::Gt => ">",
                    CmpOp::Ge => ">=",
                    CmpOp::Like => "LIKE",
                    CmpOp::NotLike => "NOT LIKE",
                    _ => unreachable!(),
                };
                Ok(SqliteWhere {
                    clause: format!("json_extract(meta, ?) {sql_op} ?"),
                    params: vec![path, sqlite_value(value)?],
                })
            }
            CmpOp::In | CmpOp::NotIn => {
                let FilterValue::List(items) = value else {
                    bail!("{} requires a list value", op.sql());
                };
                if items.is_empty() {
                    return Ok(SqliteWhere {
                        clause: if op == CmpOp::In {
                            "(0 = 1)".to_string()
                        } else {
                            "(1 = 1)".to_string()
                        },
                        params: Vec::new(),
                    });
                }

                let placeholders = vec!["?"; items.len()].join(", ");
                let sql_op = if op == CmpOp::In { "IN" } else { "NOT IN" };
                let params = items.iter().map(sqlite_value).collect::<Result<Vec<_>>>()?;
                let mut all_params = Vec::with_capacity(params.len() + 1);
                all_params.push(path);
                all_params.extend(params);
                Ok(SqliteWhere {
                    clause: format!("json_extract(meta, ?) {sql_op} ({placeholders})"),
                    params: all_params,
                })
            }
        }
    }
}

trait ResultExt<T> {
    fn context_filter(self, msg: &str) -> Result<T>;
}

impl<T, E> ResultExt<T> for std::result::Result<T, E>
where
    E: std::fmt::Display,
{
    fn context_filter(self, msg: &str) -> Result<T> {
        self.map_err(|err| anyhow!("{msg}: {err}"))
    }
}

#[cfg(test)]
#[path = "filter_test.rs"]
mod tests;
