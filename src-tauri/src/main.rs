#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use base64::{engine::general_purpose, Engine as _};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf, process::Command};
#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GraphOptions {
    input_path: String,
    min_term_freq: u32,
    edge_threshold: f64,
    min_degree: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GraphResult {
    output_path: String,
    image_data_url: String,
}

#[tauri::command]
fn save_graph_image(source_path: String, destination_path: String) -> Result<(), String> {
    let source = PathBuf::from(source_path);
    let destination = PathBuf::from(destination_path);

    if source.extension().and_then(|extension| extension.to_str()) != Some("png") {
        return Err("저장할 그래프 이미지가 없습니다. 그래프를 먼저 생성하세요.".into());
    }
    if !source.exists() {
        return Err("그래프 이미지가 더 이상 존재하지 않습니다. 다시 생성해 주세요.".into());
    }

    fs::copy(&source, &destination)
        .map_err(|error| format!("이미지 파일을 저장하지 못했습니다: {error}"))?;
    Ok(())
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri must be inside the workspace")
        .to_path_buf()
}

fn application_root() -> PathBuf {
    // 포터블 배포본은 실행 파일 옆의 분석기와 Python 런타임을 사용한다.
    if let Ok(executable) = std::env::current_exe() {
        if let Some(directory) = executable.parent() {
            if directory.join("graph_maker.py").exists()
                && directory.join("python").join("python.exe").exists()
            {
                return directory.to_path_buf();
            }
        }
    }
    // 개발 환경에서는 기존 프로젝트 가상환경을 계속 사용한다.
    workspace_root()
}

#[cfg(not(debug_assertions))]
const EMBEDDED_ANALYZER: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/resources/graph_maker.exe"
));

#[cfg(not(debug_assertions))]
fn bundled_analyzer_path() -> Result<PathBuf, String> {
    // 최종 단일 EXE에는 분석기도 리소스로 포함한다. 처음 분석할 때만 임시 폴더에 풀어 쓴다.
    let runtime_dir = std::env::temp_dir().join("ConTextation").join("runtime");
    fs::create_dir_all(&runtime_dir)
        .map_err(|error| format!("분석기 임시 폴더를 만들지 못했습니다: {error}"))?;

    let analyzer = runtime_dir.join("graph_maker.exe");
    let should_extract = fs::metadata(&analyzer)
        .map(|metadata| metadata.len() != EMBEDDED_ANALYZER.len() as u64)
        .unwrap_or(true);

    if should_extract {
        fs::write(&analyzer, EMBEDDED_ANALYZER)
            .map_err(|error| format!("내장 분석기를 준비하지 못했습니다: {error}"))?;
    }

    Ok(analyzer)
}

#[tauri::command]
fn generate_graph(options: GraphOptions) -> Result<GraphResult, String> {
    #[cfg(debug_assertions)]
    let root = application_root();

    #[cfg(debug_assertions)]
    let portable_python = root.join("python").join("python.exe");

    #[cfg(debug_assertions)]
    let python = if portable_python.exists() {
        portable_python
    } else {
        root.join("venv").join("Scripts").join("python.exe")
    };

    #[cfg(debug_assertions)]
    let script = root.join("graph_maker.py");

    #[cfg(debug_assertions)]
    let output = root.join("knowledge_graph_ui.png");

    #[cfg(not(debug_assertions))]
    let analyzer = bundled_analyzer_path()?;

    #[cfg(not(debug_assertions))]
    let output = std::env::temp_dir()
        .join("ConTextation")
        .join(format!("knowledge_graph_{}.png", std::process::id()));

    #[cfg(debug_assertions)]
    if !python.exists() {
        return Err("Python 분석 환경을 찾지 못했습니다. 포터블 앱 폴더의 python 디렉터리를 확인하세요.".into());
    }

    #[cfg(debug_assertions)]
    if !script.exists() {
        return Err("graph_maker.py를 찾지 못했습니다.".into());
    }

    #[cfg(debug_assertions)]
    let mut command = Command::new(python);

    #[cfg(not(debug_assertions))]
    let mut command = Command::new(analyzer);

    #[cfg(windows)]
    command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW

    #[cfg(debug_assertions)]
    command.arg(script);

    let command_output = command
        .arg(&options.input_path)
        .args(["--min-term-freq", &options.min_term_freq.to_string()])
        .args(["--edge-threshold", &options.edge_threshold.to_string()])
        .args(["--min-degree", &options.min_degree.to_string()])
        .args(["--output", output.to_string_lossy().as_ref()])
        .output()
        .map_err(|error| format!("분석기를 실행하지 못했습니다: {error}"))?;

    if !command_output.status.success() {
        let stderr = String::from_utf8_lossy(&command_output.stderr);
        let stdout = String::from_utf8_lossy(&command_output.stdout);
        return Err(format!("그래프 생성에 실패했습니다.\n{stderr}\n{stdout}"));
    }

    let image = fs::read(&output).map_err(|error| format!("생성된 그래프를 읽지 못했습니다: {error}"))?;
    Ok(GraphResult {
        output_path: output.to_string_lossy().to_string(),
        image_data_url: format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(image)),
    })
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![generate_graph, save_graph_image])
        .run(tauri::generate_context!())
        .expect("error while running ConTextation");
}
