use std::{borrow::Cow, collections::HashMap, fs, path::Path, time::Duration};

use anyhow::{anyhow, Context, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivEnvironment {
    Demo,
    Real,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Gaussian,
    Transformer,
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
    pub trading_enabled: bool,
    pub model_type: ModelType,
    pub transformer_model_path: Option<String>,
    pub transformer_sequence_length: usize,
    /// Minimum number of ticks a trend must persist (momentum guard).
    pub min_trend_length: u32,
    /// Added to threshold when volatility < 0.0001.
    pub strategy_volatility_penalty: f64,
    /// Subtracted from threshold when streak >= 4.
    pub strategy_momentum_reward: f64,
    /// Minimum return as ratio of volatility (noise filter).
    pub strategy_min_return_ratio: f64,
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        let env_map = load_dotenv(".env")?;

        let deriv_environment_str = lookup_env(&env_map, "DERIV_ENVIRONMENT")
            .unwrap_or(Cow::Borrowed("DEMO"))
            .to_ascii_uppercase();

        let deriv_environment = match deriv_environment_str.as_str() {
            "DEMO" => DerivEnvironment::Demo,
            "REAL" => DerivEnvironment::Real,
            other => return Err(anyhow!("unsupported DERIV_ENVIRONMENT value: {other}")),
        };

        let token = match deriv_environment {
            DerivEnvironment::Demo => lookup_env(&env_map, "DERIV_DEMO_TOKEN")?.into_owned(),
            DerivEnvironment::Real => lookup_env(&env_map, "DERIV_REAL_TOKEN")?.into_owned(),
        };

        Ok(Self {
            websocket_endpoint: lookup_env(&env_map, "DERIV_WS_ENDPOINT")
                .map(|v| v.into_owned())
                .unwrap_or_else(|_| "wss://ws.derivws.com/websockets/v3".to_string()),
            app_id: parse_or_default(&env_map, "DERIV_APP_ID", 1089)?,
            deriv_environment,
            token,
            symbol: lookup_env(&env_map, "DERIV_SYMBOL")
                .map(|v| v.into_owned())
                .unwrap_or_else(|_| "R_100".to_string()),
            contract_type: lookup_env(&env_map, "DERIV_CONTRACT_TYPE")
                .map(|v| v.into_owned())
                .unwrap_or_else(|_| "CALL".to_string()),
            currency: lookup_env(&env_map, "DERIV_CURRENCY")
                .map(|v| v.into_owned())
                .unwrap_or_else(|_| "USD".to_string()),
            stake: parse_or_default(&env_map, "DERIV_STAKE", 1.0)?,
            duration_ticks: parse_or_default(&env_map, "DERIV_DURATION_TICKS", 1)?,
            probability_threshold: parse_or_default(&env_map, "DERIV_THRESHOLD", 0.55)?,
            reconnect_backoff: Duration::from_millis(parse_or_default(
                &env_map,
                "DERIV_RECONNECT_BACKOFF_MS",
                1_000_u64,
            )?),
            min_api_interval: Duration::from_millis(parse_or_default(
                &env_map,
                "DERIV_MIN_API_INTERVAL_MS",
                250_u64,
            )?),
            cooldown_ticks: parse_or_default(&env_map, "DERIV_COOLDOWN_TICKS", 16)?,
            max_tick_latency: Duration::from_millis(parse_or_default(
                &env_map,
                "DERIV_MAX_TICK_LATENCY_MS",
                10_u64,
            )?),
            inbound_queue_capacity: parse_or_default(
                &env_map,
                "DERIV_INBOUND_QUEUE_CAPACITY",
                1024_usize,
            )?,
            outbound_queue_capacity: parse_or_default(
                &env_map,
                "DERIV_OUTBOUND_QUEUE_CAPACITY",
                128_usize,
            )?,
            trading_enabled: parse_or_default(&env_map, "DERIV_TRADING_ENABLED", false)?,
            model_type: match lookup_env(&env_map, "MODEL_TYPE")
                .unwrap_or(Cow::Borrowed("GAUSSIAN"))
                .to_ascii_uppercase()
                .as_str()
            {
                "TRANSFORMER" => ModelType::Transformer,
                _ => ModelType::Gaussian,
            },
            transformer_model_path: lookup_env(&env_map, "TRANSFORMER_MODEL_PATH")
                .map(|v| v.into_owned())
                .ok(),
            transformer_sequence_length: parse_or_default(
                &env_map,
                "TRANSFORMER_SEQUENCE_LENGTH",
                32_usize,
            )?,
            min_trend_length: parse_or_default(&env_map, "STRATEGY_MIN_TREND_LENGTH", 5)?,
            strategy_volatility_penalty: parse_or_default(
                &env_map,
                "STRATEGY_VOLATILITY_PENALTY",
                0.05,
            )?,
            strategy_momentum_reward: parse_or_default(&env_map, "STRATEGY_MOMENTUM_REWARD", 0.02)?,
            strategy_min_return_ratio: parse_or_default(
                &env_map,
                "STRATEGY_MIN_RETURN_RATIO",
                0.1,
            )?,
        })
    }
}

fn lookup_env<'a>(env_map: &'a HashMap<String, String>, key: &str) -> Result<Cow<'a, str>> {
    if let Ok(val) = std::env::var(key) {
        Ok(Cow::Owned(val))
    } else if let Some(value) = env_map.get(key) {
        Ok(Cow::Borrowed(value))
    } else {
        Err(anyhow!("missing required environment variable {key}"))
    }
}

fn parse_or_default<T>(env_map: &HashMap<String, String>, key: &str, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match lookup_env(env_map, key) {
        Ok(value) => value
            .parse::<T>()
            .map_err(|err| anyhow!("failed to parse {key}: {err}")),
        Err(_) => Ok(default),
    }
}

fn load_dotenv(path: &str) -> Result<HashMap<String, String>> {
    let mut env_map = HashMap::new();
    let dotenv_path = Path::new(path);
    if !dotenv_path.exists() {
        return Ok(env_map);
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
        env_map.insert(key.trim().to_string(), cleaned.to_string());
    }

    Ok(env_map)
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
