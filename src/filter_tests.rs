use super::*;
use filterql::Expr;

#[test]
fn filter_compiler_uses_placeholders_and_params() {
    let compiled =
        compile_filter("source = 'docs' AND lang = 'zh' AND kind IN ('guide','api')").unwrap();
    assert!(compiled.clause.contains("json_extract(meta, ?) = ?"));
    assert!(compiled.clause.contains("json_extract(meta, ?) IN (?, ?)"));
    assert_eq!(compiled.params.len(), 7);
}

#[test]
fn filter_compiler_rejects_invalid_field_names() {
    let err = compile_filter("bad-field = 'x'").unwrap_err().to_string();
    assert!(err.contains("unsupported characters") || err.contains("parsing metadata filter"));
}

#[test]
fn empty_in_lists_have_total_boolean_semantics() {
    let in_clause = filterql::compile(
        &Expr::cmp("source", CmpOp::In, FilterValue::List(Vec::new())),
        &mut SqliteFilterCompiler,
    )
    .unwrap();
    assert_eq!(in_clause.clause, "(0 = 1)");

    let not_in_clause = filterql::compile(
        &Expr::cmp("source", CmpOp::NotIn, FilterValue::List(Vec::new())),
        &mut SqliteFilterCompiler,
    )
    .unwrap();
    assert_eq!(not_in_clause.clause, "(1 = 1)");
}
