use std::{env, fs, path::Path, time::Duration};

use anyhow::{anyhow, Context, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivEnvironment {
    Demo,
    Real,
}

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub websocket_endpoint: String,
    pub app_id: u32,
    pub deriv_environment: DerivEnvironment,
    pub token: String,
    pub symbol: String,
    pub contract_type: String,
    pub currency: String,
    pub stake: f64,
    pub duration_ticks: u32,
    pub probability_threshold: f64,
    pub reconnect_backoff: Duration,
    pub min_api_interval: Duration,
    pub cooldown_ticks: u32,
    pub max_tick_latency: Duration,
    pub inbound_queue_capacity: usize,
    pub outbound_queue_capacity: usize,
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        load_dotenv(".env")?;

        let deriv_environment = match env_var("DERIV_ENVIRONMENT")
            .unwrap_or_else(|_| "DEMO".to_string())
            .to_ascii_uppercase()
            .as_str()
        {
            "DEMO" => DerivEnvironment::Demo,
            "REAL" => DerivEnvironment::Real,
            other => return Err(anyhow!("unsupported DERIV_ENVIRONMENT value: {other}")),
        };

        let token = match deriv_environment {
            DerivEnvironment::Demo => env_var("DERIV_DEMO_TOKEN")?,
            DerivEnvironment::Real => env_var("DERIV_REAL_TOKEN")?,
        };

        Ok(Self {
            websocket_endpoint: env::var("DERIV_WS_ENDPOINT")
                .unwrap_or_else(|_| "wss://ws.derivws.com/websockets/v3".to_string()),
            app_id: parse_or_default("DERIV_APP_ID", 1089)?,
            deriv_environment,
            token,
            symbol: env::var("DERIV_SYMBOL").unwrap_or_else(|_| "R_100".to_string()),
            contract_type: env::var("DERIV_CONTRACT_TYPE").unwrap_or_else(|_| "CALL".to_string()),
            currency: env::var("DERIV_CURRENCY").unwrap_or_else(|_| "USD".to_string()),
            stake: parse_or_default("DERIV_STAKE", 1.0)?,
            duration_ticks: parse_or_default("DERIV_DURATION_TICKS", 5)?,
            probability_threshold: parse_or_default("DERIV_THRESHOLD", 0.55)?,
            reconnect_backoff: Duration::from_millis(parse_or_default(
                "DERIV_RECONNECT_BACKOFF_MS",
                1_000_u64,
            )?),
            min_api_interval: Duration::from_millis(parse_or_default(
                "DERIV_MIN_API_INTERVAL_MS",
                250_u64,
            )?),
            cooldown_ticks: parse_or_default("DERIV_COOLDOWN_TICKS", 16)?,
            max_tick_latency: Duration::from_millis(parse_or_default(
                "DERIV_MAX_TICK_LATENCY_MS",
                10_u64,
            )?),
            inbound_queue_capacity: parse_or_default("DERIV_INBOUND_QUEUE_CAPACITY", 1024_usize)?,
            outbound_queue_capacity: parse_or_default("DERIV_OUTBOUND_QUEUE_CAPACITY", 128_usize)?,
        })
    }
}

fn env_var(key: &str) -> Result<String> {
    env::var(key).with_context(|| format!("missing required environment variable {key}"))
}

fn parse_or_default<T>(key: &str, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match env::var(key) {
        Ok(value) => value
            .parse::<T>()
            .map_err(|err| anyhow!("failed to parse {key}: {err}")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(err) => Err(anyhow!("failed to load {key}: {err}")),
    }
}

fn load_dotenv(path: &str) -> Result<()> {
    let dotenv_path = Path::new(path);
    if !dotenv_path.exists() {
        return Ok(());
    }

    let contents = fs::read_to_string(dotenv_path)
        .with_context(|| format!("failed to read {}", dotenv_path.display()))?;

    for raw_line in contents.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let Some((key, value)) = line.split_once('=') else {
            continue;
        };

        let cleaned = value.trim().trim_matches('"').trim_matches('\'');
        if env::var_os(key.trim()).is_none() {
            env::set_var(key.trim(), cleaned);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_environment_case_insensitively() {
        std::env::set_var("DERIV_ENVIRONMENT", "real");
        std::env::set_var("DERIV_REAL_TOKEN", "token");

        let config = AppConfig::load().expect("config should load");

        assert_eq!(config.deriv_environment, DerivEnvironment::Real);
        assert_eq!(config.token, "token");
    }
}
