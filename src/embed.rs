//! Shared embedding helpers (client construction, vector utilities).

use anyhow::{bail, Result};
use embeddings::{resolve_api_key_for_provider, EmbedClient};

use crate::config::normalize_provider_model;

pub(crate) fn build_client(
    model: &str,
    provider: Option<&str>,
    base_url: Option<&str>,
) -> Result<EmbedClient> {
    let (model, provider) = normalize_provider_model(model, provider)?;
    let mut client = if let Some(p) = provider.as_deref() {
        let api_key = resolve_api_key_for_provider(p)?;
        EmbedClient::new(p, &model, api_key)?
    } else {
        EmbedClient::from_env(&model)?
    };
    if let Some(url) = base_url {
        client = client.with_base_url(url);
    }
    Ok(client)
}

pub(crate) fn flatten_embeddings(embeddings: &[Vec<f32>]) -> Vec<f32> {
    let mut flat = Vec::with_capacity(embeddings.len() * embeddings.first().map_or(0, |v| v.len()));
    for emb in embeddings {
        flat.extend_from_slice(emb);
    }
    flat
}

pub(crate) fn validate_vectors_dim(vectors: &[Vec<f32>], dim: usize) -> Result<()> {
    for (idx, vector) in vectors.iter().enumerate() {
        if vector.len() != dim {
            bail!(
                "vector dimension mismatch at batch item {}: index expects {}, got {}",
                idx,
                dim,
                vector.len()
            );
        }
    }
    Ok(())
}
