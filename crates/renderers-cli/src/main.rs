//! `renderers-cli` — small dev tool that drives `renderers-core`
//! without going through Python.
//!
//! Designed for two use cases:
//!
//! 1. **Golden parity checking**: render a fixture JSON of messages
//!    against a tokenizer.json, emit the result as JSON, and `diff`
//!    against the Python reference output. The exit code is non-zero
//!    if the run fails — the comparison is left to the caller (the
//!    pytest harness does the actual diffing).
//! 2. **Manual prototyping**: try out new families / config changes
//!    without spinning up the PyO3 wheel.
//!
//! Usage:
//!
//! ```text
//! renderers-cli render --family qwen3 --tokenizer tokenizer.json \
//!     --messages conversation.json [--tools tools.json] [--gen-prompt]
//!
//! renderers-cli parse  --family qwen3 --tokenizer tokenizer.json \
//!     --token-ids '[151644, 8948, ...]'
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use renderers_core::families::Qwen3Renderer;
use renderers_core::tokenizer::Tokenizer;
use renderers_core::types::{Message, ParsedToolCall, RenderedTokens, ToolArguments, ToolSpec};
use renderers_core::Renderer;
use serde::Serialize;

fn print_usage() {
    eprintln!(
        "renderers-cli — render and parse via renderers-core\n\n\
         USAGE:\n\
           renderers-cli render --family qwen3 --tokenizer <path> --messages <path> [--tools <path>] [--gen-prompt]\n\
           renderers-cli parse  --family qwen3 --tokenizer <path> --token-ids <json>\n\n\
         Output is line-by-line JSON on stdout.\n"
    );
}

#[derive(Serialize)]
struct RenderedJson {
    token_ids: Vec<u32>,
    message_indices: Vec<i32>,
}

impl From<RenderedTokens> for RenderedJson {
    fn from(r: RenderedTokens) -> Self {
        Self {
            token_ids: r.token_ids,
            message_indices: r.message_indices,
        }
    }
}

#[derive(Serialize)]
struct ParsedToolCallJson<'a> {
    raw: &'a str,
    name: Option<&'a str>,
    arguments: serde_json::Value,
    status: &'static str,
    token_span: Option<(usize, usize)>,
    id: Option<&'a str>,
}

impl<'a> From<&'a ParsedToolCall> for ParsedToolCallJson<'a> {
    fn from(p: &'a ParsedToolCall) -> Self {
        let args = match &p.arguments {
            None => serde_json::Value::Null,
            Some(ToolArguments::Object(v)) => v.clone(),
            Some(ToolArguments::Raw(s)) => serde_json::Value::String(s.clone()),
        };
        Self {
            raw: &p.raw,
            name: p.name.as_deref(),
            arguments: args,
            status: p.status.as_wire(),
            token_span: p.token_span.as_ref().map(|r| (r.start, r.end)),
            id: p.id.as_deref(),
        }
    }
}

#[derive(Serialize)]
struct ParsedJson<'a> {
    content: &'a str,
    reasoning_content: Option<&'a str>,
    tool_calls: Vec<ParsedToolCallJson<'a>>,
}

struct Args {
    cmd: Cmd,
    family: String,
    tokenizer: PathBuf,
    messages: Option<PathBuf>,
    tools: Option<PathBuf>,
    token_ids: Option<String>,
    gen_prompt: bool,
}

enum Cmd {
    Render,
    Parse,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let cmd = match it.next().as_deref() {
        Some("render") => Cmd::Render,
        Some("parse") => Cmd::Parse,
        Some(other) => return Err(format!("unknown command: {other}")),
        None => return Err("missing command".to_string()),
    };
    let mut args = Args {
        cmd,
        family: "qwen3".to_string(),
        tokenizer: PathBuf::new(),
        messages: None,
        tools: None,
        token_ids: None,
        gen_prompt: false,
    };
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--family" => args.family = it.next().ok_or("missing --family value")?,
            "--tokenizer" => args.tokenizer = it.next().ok_or("missing --tokenizer value")?.into(),
            "--messages" => args.messages = Some(it.next().ok_or("missing --messages value")?.into()),
            "--tools" => args.tools = Some(it.next().ok_or("missing --tools value")?.into()),
            "--token-ids" => args.token_ids = Some(it.next().ok_or("missing --token-ids value")?),
            "--gen-prompt" => args.gen_prompt = true,
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    Ok(args)
}

fn build_renderer(
    family: &str,
    tokenizer: Tokenizer,
) -> Result<Box<dyn Renderer>, String> {
    match family {
        "qwen3" => Qwen3Renderer::new(tokenizer)
            .map(|r| Box::new(r) as Box<dyn Renderer>)
            .map_err(|e| e.to_string()),
        other => Err(format!("unsupported family: {other}")),
    }
}

fn load_messages(path: &PathBuf) -> Result<Vec<Message>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("messages JSON: {e}"))
}

fn load_tools(path: &PathBuf) -> Result<Vec<ToolSpec>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("tools JSON: {e}"))
}

fn parse_token_ids(s: &str) -> Result<Vec<u32>, String> {
    let v: Vec<i64> = serde_json::from_str(s).map_err(|e| format!("token-ids JSON: {e}"))?;
    v.into_iter()
        .map(|t| {
            if !(0..=u32::MAX as i64).contains(&t) {
                Err(format!("token id out of range: {t}"))
            } else {
                Ok(t as u32)
            }
        })
        .collect()
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let tok = Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| format!("load tokenizer {:?}: {e}", args.tokenizer))?;
    let renderer = build_renderer(&args.family, tok)?;

    match args.cmd {
        Cmd::Render => {
            let messages = load_messages(
                args.messages
                    .as_ref()
                    .ok_or("--messages required for render")?,
            )?;
            let tools = match args.tools.as_ref() {
                Some(p) => Some(load_tools(p)?),
                None => None,
            };
            let rendered = renderer
                .render(&messages, tools.as_deref(), args.gen_prompt)
                .map_err(|e| e.to_string())?;
            let json: RenderedJson = rendered.into();
            println!("{}", serde_json::to_string(&json).unwrap());
        }
        Cmd::Parse => {
            let ids =
                parse_token_ids(args.token_ids.as_ref().ok_or("--token-ids required for parse")?)?;
            let parsed = renderer.parse_response(&ids);
            let tool_calls: Vec<ParsedToolCallJson<'_>> =
                parsed.tool_calls.iter().map(ParsedToolCallJson::from).collect();
            let json = ParsedJson {
                content: &parsed.content,
                reasoning_content: parsed.reasoning_content.as_deref(),
                tool_calls,
            };
            println!("{}", serde_json::to_string(&json).unwrap());
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            print_usage();
            ExitCode::FAILURE
        }
    }
}
