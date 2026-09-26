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

#[test]
fn contains_compiles_to_json_each_membership() {
    let clause = filterql::compile(
        &Expr::cmp("tags", CmpOp::Contains, FilterValue::Str("md".into())),
        &mut SqliteFilterCompiler,
    )
    .unwrap();
    assert_eq!(
        clause.clause,
        "(EXISTS (SELECT 1 FROM json_each(meta, ?) AS je WHERE je.value = ?))"
    );
    assert_eq!(clause.params.len(), 2);

    let not_clause = filterql::compile(
        &Expr::cmp("tags", CmpOp::NotContains, FilterValue::Str("md".into())),
        &mut SqliteFilterCompiler,
    )
    .unwrap();
    assert!(not_clause
        .clause
        .starts_with("json_type(meta, ?) IS NOT NULL AND NOT ("));
    assert!(not_clause.clause.contains("json_each(meta, ?)"));
    assert_eq!(not_clause.params.len(), 3);
}

#[test]
fn contains_any_compiles_to_json_each_overlap() {
    let clause = filterql::compile(
        &Expr::cmp(
            "tags",
            CmpOp::ContainsAny,
            FilterValue::List(vec![
                FilterValue::Str("md".into()),
                FilterValue::Str("rs".into()),
            ]),
        ),
        &mut SqliteFilterCompiler,
    )
    .unwrap();
    assert!(clause
        .clause
        .contains("EXISTS (SELECT 1 FROM json_each(meta, ?) AS je WHERE je.value IN (?, ?))"));
    assert_eq!(clause.params.len(), 3);

    let empty = filterql::compile(
        &Expr::cmp("tags", CmpOp::ContainsAny, FilterValue::List(Vec::new())),
        &mut SqliteFilterCompiler,
    )
    .unwrap();
    assert_eq!(empty.clause, "(0 = 1)");

    let not_any = filterql::compile(
        &Expr::cmp(
            "tags",
            CmpOp::NotContainsAny,
            FilterValue::List(vec![FilterValue::Str("md".into())]),
        ),
        &mut SqliteFilterCompiler,
    )
    .unwrap();
    assert!(not_any
        .clause
        .starts_with("json_type(meta, ?) IS NOT NULL AND NOT EXISTS"));
    assert_eq!(not_any.params.len(), 3);
}
