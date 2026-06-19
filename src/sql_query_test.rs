use super::*;

#[test]
fn parses_sql_metadata_query() {
    let query = parse_sql_metadata_query(
        "SELECT pk, category FROM articles WHERE category = 'tech' LIMIT 20",
    )
    .unwrap();
    assert_eq!(query.select, vec!["pk", "category"]);
    assert_eq!(query.filter.as_deref(), Some("category = 'tech'"));
    assert_eq!(query.limit, Some(20));
}
