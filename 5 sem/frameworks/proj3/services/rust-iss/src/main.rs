use std::{collections::HashMap, time::Duration};

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use serde::Serialize;
use serde_json::Value;
use sqlx::{postgres::PgPoolOptions, PgPool, Row};
use tracing::{error, info};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[derive(Serialize)]
struct Health { status: &'static str, now: DateTime<Utc> }

#[derive(Clone)]
struct AppState {
    pool: PgPool,
    nasa_url: String,          // OSDR
    nasa_key: String,          // ключ NASA
    fallback_url: String,      // ISS where-the-iss
    every_osdr: u64,
    every_iss: u64,
    every_apod: u64,
    every_neo: u64,
    every_donki: u64,
    every_spacex: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .finish();
    let _ = tracing::subscriber::set_global_default(subscriber);

    dotenvy::dotenv().ok();

    let db_url = std::env::var("DATABASE_URL").expect("DATABASE_URL is required");

    let nasa_url = std::env::var("NASA_API_URL")
        .unwrap_or_else(|_| "https://visualization.osdr.nasa.gov/biodata/api/v2/datasets/?format=json".to_string());
    let nasa_key = std::env::var("NASA_API_KEY").unwrap_or_default();

    let fallback_url = std::env::var("WHERE_ISS_URL")
        .unwrap_or_else(|_| "https://api.wheretheiss.at/v1/satellites/25544".to_string());

    let every_osdr   = env_u64("FETCH_EVERY_SECONDS", 600);
    let every_iss    = env_u64("ISS_EVERY_SECONDS",   120);
    let every_apod   = env_u64("APOD_EVERY_SECONDS",  43200); // 12ч
    let every_neo    = env_u64("NEO_EVERY_SECONDS",   7200);  // 2ч
    let every_donki  = env_u64("DONKI_EVERY_SECONDS", 3600);  // 1ч
    let every_spacex = env_u64("SPACEX_EVERY_SECONDS",3600);

    let pool = PgPoolOptions::new().max_connections(5).connect(&db_url).await?;
    init_db(&pool).await?;

    let state = AppState {
        pool: pool.clone(),
        nasa_url: nasa_url.clone(),
        nasa_key,
        fallback_url: fallback_url.clone(),
        every_osdr, every_iss, every_apod, every_neo, every_donki, every_spacex,
    };

    // фон OSDR
    {
        let st = state.clone();
        tokio::spawn(async move {
            loop {
                if let Err(e) = fetch_and_store_osdr(&st).await { error!("osdr err {e:?}") }
                tokio::time::sleep(Duration::from_secs(st.every_osdr)).await;
            }
        });
    }
    // фон ISS
    {
        let st = state.clone();
        tokio::spawn(async move {
            loop {
                if let Err(e) = fetch_and_store_iss(&st.pool, &st.fallback_url).await { error!("iss err {e:?}") }
                tokio::time::sleep(Duration::from_secs(st.every_iss)).await;
            }
        });
    }
    // фон APOD
    {
        let st = state.clone();
        tokio::spawn(async move {
            loop {
                if let Err(e) = fetch_apod(&st).await { error!("apod err {e:?}") }
                tokio::time::sleep(Duration::from_secs(st.every_apod)).await;
            }
        });
    }
    // фон NeoWs
    {
        let st = state.clone();
        tokio::spawn(async move {
            loop {
                if let Err(e) = fetch_neo_feed(&st).await { error!("neo err {e:?}") }
                tokio::time::sleep(Duration::from_secs(st.every_neo)).await;
            }
        });
    }
    // фон DONKI
    {
        let st = state.clone();
        tokio::spawn(async move {
            loop {
                if let Err(e) = fetch_donki(&st).await { error!("donki err {e:?}") }
                tokio::time::sleep(Duration::from_secs(st.every_donki)).await;
            }
        });
    }
    // фон SpaceX
    {
        let st = state.clone();
        tokio::spawn(async move {
            loop {
                if let Err(e) = fetch_spacex_next(&st).await { error!("spacex err {e:?}") }
                tokio::time::sleep(Duration::from_secs(st.every_spacex)).await;
            }
        });
    }

    let app = Router::new()
        // общее
        .route("/health", get(|| async { Json(Health { status: "ok", now: Utc::now() }) }))
        .with_state(state.clone())
        // ISS
        .route("/last", get(last_iss))
        .route("/fetch", get(trigger_iss))
        .route("/iss/trend", get(iss_trend))
        // OSDR
        .route("/osdr/sync", get(osdr_sync))
        .route("/osdr/list", get(osdr_list))
        // Space cache
        .route("/space/:src/latest", get(space_latest))
        .route("/space/refresh", get(space_refresh))
        .route("/space/summary", get(space_summary))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", 3000)).await?;
    info!("rust_iss listening on 0.0.0.0:3000");
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}

fn env_u64(k: &str, d: u64) -> u64 {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

/* ---------- DB boot ---------- */
async fn init_db(pool: &PgPool) -> anyhow::Result<()> {
    // ISS
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS iss_fetch_log(
            id BIGSERIAL PRIMARY KEY,
            fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            source_url TEXT NOT NULL,
            payload JSONB NOT NULL
        )"
    ).execute(pool).await?;

    // OSDR
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS osdr_items(
            id BIGSERIAL PRIMARY KEY,
            dataset_id TEXT,
            title TEXT,
            status TEXT,
            updated_at TIMESTAMPTZ,
            inserted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            raw JSONB NOT NULL
        )"
    ).execute(pool).await?;
    sqlx::query(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_osdr_dataset_id
         ON osdr_items(dataset_id) WHERE dataset_id IS NOT NULL"
    ).execute(pool).await?;

    // универсальный кэш космоданных
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS space_cache(
            id BIGSERIAL PRIMARY KEY,
            source TEXT NOT NULL,
            fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            payload JSONB NOT NULL
        )"
    ).execute(pool).await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS ix_space_cache_source ON space_cache(source,fetched_at DESC)").execute(pool).await?;

    Ok(())
}

/* ---------- ISS ---------- */
async fn last_iss(State(st): State<AppState>)
-> Result<Json<Value>, (StatusCode, String)> {
    let row_opt = sqlx::query(
        "SELECT id, fetched_at, source_url, payload
         FROM iss_fetch_log
         ORDER BY id DESC LIMIT 1"
    ).fetch_optional(&st.pool).await
     .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if let Some(row) = row_opt {
        let id: i64 = row.get("id");
        let fetched_at: DateTime<Utc> = row.get::<DateTime<Utc>, _>("fetched_at");
        let source_url: String = row.get("source_url");
        let payload: Value = row.try_get("payload").unwrap_or(serde_json::json!({}));
        return Ok(Json(serde_json::json!({
            "id": id, "fetched_at": fetched_at, "source_url": source_url, "payload": payload
        })));
    }
    Ok(Json(serde_json::json!({"message":"no data"})))
}

async fn trigger_iss(State(st): State<AppState>)
-> Result<Json<Value>, (StatusCode, String)> {
    fetch_and_store_iss(&st.pool, &st.fallback_url).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    last_iss(State(st)).await
}

#[derive(Serialize)]
struct Trend {
    movement: bool,
    delta_km: f64,
    dt_sec: f64,
    velocity_kmh: Option<f64>,
    from_time: Option<DateTime<Utc>>,
    to_time: Option<DateTime<Utc>>,
    from_lat: Option<f64>,
    from_lon: Option<f64>,
    to_lat: Option<f64>,
    to_lon: Option<f64>,
}

async fn iss_trend(State(st): State<AppState>)
-> Result<Json<Trend>, (StatusCode, String)> {
    let rows = sqlx::query("SELECT fetched_at, payload FROM iss_fetch_log ORDER BY id DESC LIMIT 2")
        .fetch_all(&st.pool).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if rows.len() < 2 {
        return Ok(Json(Trend {
            movement: false, delta_km: 0.0, dt_sec: 0.0, velocity_kmh: None,
            from_time: None, to_time: None,
            from_lat: None, from_lon: None, to_lat: None, to_lon: None
        }));
    }

    let t2: DateTime<Utc> = rows[0].get("fetched_at");
    let t1: DateTime<Utc> = rows[1].get("fetched_at");
    let p2: Value = rows[0].get("payload");
    let p1: Value = rows[1].get("payload");

    let lat1 = num(&p1["latitude"]);
    let lon1 = num(&p1["longitude"]);
    let lat2 = num(&p2["latitude"]);
    let lon2 = num(&p2["longitude"]);
    let v2 = num(&p2["velocity"]);

    let mut delta_km = 0.0;
    let mut movement = false;
    if let (Some(a1), Some(o1), Some(a2), Some(o2)) = (lat1, lon1, lat2, lon2) {
        delta_km = haversine_km(a1, o1, a2, o2);
        movement = delta_km > 0.1;
    }
    let dt_sec = (t2 - t1).num_milliseconds() as f64 / 1000.0;

    Ok(Json(Trend {
        movement,
        delta_km,
        dt_sec,
        velocity_kmh: v2,
        from_time: Some(t1),
        to_time: Some(t2),
        from_lat: lat1, from_lon: lon1, to_lat: lat2, to_lon: lon2,
    }))
}

fn num(v: &Value) -> Option<f64> {
    if let Some(x) = v.as_f64() { return Some(x); }
    if let Some(s) = v.as_str() { return s.parse::<f64>().ok(); }
    None
}

fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let rlat1 = lat1.to_radians();
    let rlat2 = lat2.to_radians();
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2) + rlat1.cos() * rlat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    6371.0 * c
}

/* ---------- OSDR ---------- */
async fn osdr_sync(State(st): State<AppState>)
-> Result<Json<Value>, (StatusCode, String)> {
    let written = fetch_and_store_osdr(&st).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(serde_json::json!({ "written": written })))
}

async fn osdr_list(State(st): State<AppState>)
-> Result<Json<Value>, (StatusCode, String)> {
    let limit =  std::env::var("OSDR_LIST_LIMIT").ok()
        .and_then(|s| s.parse::<i64>().ok()).unwrap_or(20);

    let rows = sqlx::query(
        "SELECT id, dataset_id, title, status, updated_at, inserted_at, raw
         FROM osdr_items
         ORDER BY inserted_at DESC
         LIMIT $1"
    ).bind(limit).fetch_all(&st.pool).await
     .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let out: Vec<Value> = rows.into_iter().map(|r| {
        serde_json::json!({
            "id": r.get::<i64,_>("id"),
            "dataset_id": r.get::<Option<String>,_>("dataset_id"),
            "title": r.get::<Option<String>,_>("title"),
            "status": r.get::<Option<String>,_>("status"),
            "updated_at": r.get::<Option<DateTime<Utc>>,_>("updated_at"),
            "inserted_at": r.get::<DateTime<Utc>, _>("inserted_at"),
            "raw": r.get::<Value,_>("raw"),
        })
    }).collect();

    Ok(Json(serde_json::json!({ "items": out })))
}

/* ---------- Универсальная витрина space_cache ---------- */

async fn space_latest(Path(src): Path<String>, State(st): State<AppState>)
-> Result<Json<Value>, (StatusCode, String)> {
    let row = sqlx::query(
        "SELECT fetched_at, payload FROM space_cache
         WHERE source = $1 ORDER BY id DESC LIMIT 1"
    ).bind(&src).fetch_optional(&st.pool).await
     .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if let Some(r) = row {
        let fetched_at: DateTime<Utc> = r.get("fetched_at");
        let payload: Value = r.get("payload");
        return Ok(Json(serde_json::json!({ "source": src, "fetched_at": fetched_at, "payload": payload })));
    }
    Ok(Json(serde_json::json!({ "source": src, "message":"no data" })))
}

async fn space_refresh(Query(q): Query<HashMap<String,String>>, State(st): State<AppState>)
-> Result<Json<Value>, (StatusCode, String)> {
    let list = q.get("src").cloned().unwrap_or_else(|| "apod,neo,flr,cme,spacex".to_string());
    let mut done = Vec::new();
    for s in list.split(',').map(|x| x.trim().to_lowercase()) {
        match s.as_str() {
            "apod"   => { let _ = fetch_apod(&st).await;       done.push("apod"); }
            "neo"    => { let _ = fetch_neo_feed(&st).await;   done.push("neo"); }
            "flr"    => { let _ = fetch_donki_flr(&st).await;  done.push("flr"); }
            "cme"    => { let _ = fetch_donki_cme(&st).await;  done.push("cme"); }
            "spacex" => { let _ = fetch_spacex_next(&st).await; done.push("spacex"); }
            _ => {}
        }
    }
    Ok(Json(serde_json::json!({ "refreshed": done })))
}

async fn latest_from_cache(pool: &PgPool, src: &str) -> Value {
    sqlx::query("SELECT fetched_at, payload FROM space_cache WHERE source=$1 ORDER BY id DESC LIMIT 1")
        .bind(src)
        .fetch_optional(pool).await.ok().flatten()
        .map(|r| serde_json::json!({"at": r.get::<DateTime<Utc>,_>("fetched_at"), "payload": r.get::<Value,_>("payload")}))
        .unwrap_or(serde_json::json!({}))
}

async fn space_summary(State(st): State<AppState>)
-> Result<Json<Value>, (StatusCode, String)> {
    let apod   = latest_from_cache(&st.pool, "apod").await;
    let neo    = latest_from_cache(&st.pool, "neo").await;
    let flr    = latest_from_cache(&st.pool, "flr").await;
    let cme    = latest_from_cache(&st.pool, "cme").await;
    let spacex = latest_from_cache(&st.pool, "spacex").await;

    let iss_last = sqlx::query("SELECT fetched_at,payload FROM iss_fetch_log ORDER BY id DESC LIMIT 1")
        .fetch_optional(&st.pool).await.ok().flatten()
        .map(|r| serde_json::json!({"at": r.get::<DateTime<Utc>,_>("fetched_at"), "payload": r.get::<Value,_>("payload")}))
        .unwrap_or(serde_json::json!({}));

    let osdr_count: i64 = sqlx::query("SELECT count(*) AS c FROM osdr_items")
        .fetch_one(&st.pool).await.map(|r| r.get::<i64,_>("c")).unwrap_or(0);

    Ok(Json(serde_json::json!({
        "apod": apod, "neo": neo, "flr": flr, "cme": cme, "spacex": spacex,
        "iss": iss_last, "osdr_count": osdr_count
    })))
}

/* ---------- Фетчеры и запись ---------- */

async fn write_cache(pool: &PgPool, source: &str, payload: Value) -> anyhow::Result<()> {
    sqlx::query("INSERT INTO space_cache(source, payload) VALUES ($1,$2)")
        .bind(source).bind(payload).execute(pool).await?;
    Ok(())
}

// APOD
async fn fetch_apod(st: &AppState) -> anyhow::Result<()> {
    let url = "https://api.nasa.gov/planetary/apod";
    let client = reqwest::Client::builder().timeout(Duration::from_secs(30)).build()?;
    let mut req = client.get(url).query(&[("thumbs","true")]);
    if !st.nasa_key.is_empty() { req = req.query(&[("api_key",&st.nasa_key)]); }
    let json: Value = req.send().await?.json().await?;
    write_cache(&st.pool, "apod", json).await
}

// NeoWs
async fn fetch_neo_feed(st: &AppState) -> anyhow::Result<()> {
    let today = Utc::now().date_naive();
    let start = today - chrono::Days::new(2);
    let url = "https://api.nasa.gov/neo/rest/v1/feed";
    let client = reqwest::Client::builder().timeout(Duration::from_secs(30)).build()?;
    let mut req = client.get(url).query(&[
        ("start_date", start.to_string()),
        ("end_date", today.to_string()),
    ]);
    if !st.nasa_key.is_empty() { req = req.query(&[("api_key",&st.nasa_key)]); }
    let json: Value = req.send().await?.json().await?;
    write_cache(&st.pool, "neo", json).await
}

// DONKI объединённая
async fn fetch_donki(st: &AppState) -> anyhow::Result<()> {
    let _ = fetch_donki_flr(st).await;
    let _ = fetch_donki_cme(st).await;
    Ok(())
}
async fn fetch_donki_flr(st: &AppState) -> anyhow::Result<()> {
    let (from,to) = last_days(5);
    let url = "https://api.nasa.gov/DONKI/FLR";
    let client = reqwest::Client::builder().timeout(Duration::from_secs(30)).build()?;
    let mut req = client.get(url).query(&[("startDate",from),("endDate",to)]);
    if !st.nasa_key.is_empty() { req = req.query(&[("api_key",&st.nasa_key)]); }
    let json: Value = req.send().await?.json().await?;
    write_cache(&st.pool, "flr", json).await
}
async fn fetch_donki_cme(st: &AppState) -> anyhow::Result<()> {
    let (from,to) = last_days(5);
    let url = "https://api.nasa.gov/DONKI/CME";
    let client = reqwest::Client::builder().timeout(Duration::from_secs(30)).build()?;
    let mut req = client.get(url).query(&[("startDate",from),("endDate",to)]);
    if !st.nasa_key.is_empty() { req = req.query(&[("api_key",&st.nasa_key)]); }
    let json: Value = req.send().await?.json().await?;
    write_cache(&st.pool, "cme", json).await
}

// SpaceX
async fn fetch_spacex_next(st: &AppState) -> anyhow::Result<()> {
    let url = "https://api.spacexdata.com/v4/launches/next";
    let client = reqwest::Client::builder().timeout(Duration::from_secs(30)).build()?;
    let json: Value = client.get(url).send().await?.json().await?;
    write_cache(&st.pool, "spacex", json).await
}

fn last_days(n: i64) -> (String,String) {
    let to = Utc::now().date_naive();
    let from = to - chrono::Days::new(n as u64);
    (from.to_string(), to.to_string())
}

/* ---------- Вспомогательное ---------- */
fn s_pick(v: &Value, keys: &[&str]) -> Option<String> {
    for k in keys {
        if let Some(x) = v.get(*k) {
            if let Some(s) = x.as_str() { if !s.is_empty() { return Some(s.to_string()); } }
            else if x.is_number() { return Some(x.to_string()); }
        }
    }
    None
}
fn t_pick(v: &Value, keys: &[&str]) -> Option<DateTime<Utc>> {
    for k in keys {
        if let Some(x) = v.get(*k) {
            if let Some(s) = x.as_str() {
                if let Ok(dt) = s.parse::<DateTime<Utc>>() { return Some(dt); }
                if let Ok(ndt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
                    return Some(Utc.from_utc_datetime(&ndt));
                }
            } else if let Some(n) = x.as_i64() {
                return Some(Utc.timestamp_opt(n, 0).single().unwrap_or_else(Utc::now));
            }
        }
    }
    None
}
async fn fetch_and_store_iss(pool: &PgPool, url: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::builder().timeout(Duration::from_secs(20)).build()?;
    let resp = client.get(url).send().await?;
    let json: Value = resp.json().await?;
    sqlx::query("INSERT INTO iss_fetch_log (source_url, payload) VALUES ($1, $2)")
        .bind(url).bind(json).execute(pool).await?;
    Ok(())
}
async fn fetch_and_store_osdr(st: &AppState) -> anyhow::Result<usize> {
    let client = reqwest::Client::builder().timeout(Duration::from_secs(30)).build()?;
    let resp = client.get(&st.nasa_url).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("OSDR request status {}", resp.status());
    }
    let json: Value = resp.json().await?;
    let items = if let Some(a) = json.as_array() { a.clone() }
        else if let Some(v) = json.get("items").and_then(|x| x.as_array()) { v.clone() }
        else if let Some(v) = json.get("results").and_then(|x| x.as_array()) { v.clone() }
        else { vec![json.clone()] };

    let mut written = 0usize;
    for item in items {
        let id = s_pick(&item, &["dataset_id","id","uuid","studyId","accession","osdr_id"]);
        let title = s_pick(&item, &["title","name","label"]);
        let status = s_pick(&item, &["status","state","lifecycle"]);
        let updated = t_pick(&item, &["updated","updated_at","modified","lastUpdated","timestamp"]);
        if let Some(ds) = id.clone() {
            sqlx::query(
                "INSERT INTO osdr_items(dataset_id, title, status, updated_at, raw)
                 VALUES($1,$2,$3,$4,$5)
                 ON CONFLICT (dataset_id) DO UPDATE
                 SET title=EXCLUDED.title, status=EXCLUDED.status,
                     updated_at=EXCLUDED.updated_at, raw=EXCLUDED.raw"
            ).bind(ds).bind(title).bind(status).bind(updated).bind(item).execute(&st.pool).await?;
        } else {
            sqlx::query(
                "INSERT INTO osdr_items(dataset_id, title, status, updated_at, raw)
                 VALUES($1,$2,$3,$4,$5)"
            ).bind::<Option<String>>(None).bind(title).bind(status).bind(updated).bind(item).execute(&st.pool).await?;
        }
        written += 1;
    }
    Ok(written)
}
