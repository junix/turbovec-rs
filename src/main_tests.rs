use super::*;
use std::path::PathBuf;

#[test]
fn cli_parses_filter_ids_subcommand() {
    let cli = Cli::parse_from([
        "turbovec-rs",
        "filter-ids",
        "--db",
        "/tmp/docs.tvim",
        "--filter",
        "lang = 'zh'",
    ]);

    match cli.command {
        Commands::FilterIds { db, filter } => {
            assert_eq!(db, Some(PathBuf::from("/tmp/docs.tvim")));
            assert_eq!(filter, "lang = 'zh'");
        }
        _ => panic!("expected filter-ids subcommand"),
    }
}

#[test]
fn cli_parses_describe_without_db() {
    let cli = Cli::parse_from(["turbovec-rs", "describe"]);
    assert!(matches!(cli.command, Commands::Describe));
}

#[test]
fn cli_accepts_dryrun_aliases() {
    let dryrun = Cli::parse_from(["turbovec-rs", "--dryrun", "stats"]);
    let dry_run = Cli::parse_from(["turbovec-rs", "--dry-run", "stats"]);
    assert!(dryrun.dryrun);
    assert!(dry_run.dryrun);
}

#[test]
fn cli_parses_search_vector_without_query() {
    let cli = Cli::parse_from([
        "turbovec-rs",
        "search",
        "--db",
        "/tmp/docs.tvim",
        "--vector",
        "[0.1,0.2]",
    ]);

    match cli.command {
        Commands::Search { query, vector, .. } => {
            assert!(query.is_none());
            assert_eq!(vector.as_deref(), Some("[0.1,0.2]"));
        }
        _ => panic!("expected search subcommand"),
    }
}
