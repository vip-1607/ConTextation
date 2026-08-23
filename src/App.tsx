import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";

type GraphResult = {
  outputPath: string;
  imageDataUrl: string;
};

const fileName = (path: string) => path.split(/[\\/]/).at(-1) ?? path;
const graphFileName = (path: string) => {
  const name = fileName(path).replace(/\.[^.]+$/, "");
  return `${name}_지식그래프.png`;
};

export default function App() {
  const [inputPath, setInputPath] = useState("");
  const [minTermFreq, setMinTermFreq] = useState(2);
  const [edgeThreshold, setEdgeThreshold] = useState(0.03);
  const [minDegree, setMinDegree] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<GraphResult | null>(null);

  const chooseFile = async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "문서", extensions: ["txt", "md", "pdf"] }],
    });
    if (typeof selected === "string") {
      setInputPath(selected);
      setError("");
      setResult(null);
    }
  };

  const generateGraph = async () => {
    if (!inputPath) return;
    setLoading(true);
    setError("");
    try {
      const nextResult = await invoke<GraphResult>("generate_graph", {
        options: { inputPath, minTermFreq, edgeThreshold, minDegree },
      });
      setResult(nextResult);
    } catch (reason) {
      setError(String(reason));
    } finally {
      setLoading(false);
    }
  };

  const saveGraphImage = async () => {
    if (!result) return;
    const destinationPath = await save({
      defaultPath: graphFileName(inputPath),
      filters: [{ name: "PNG 이미지", extensions: ["png"] }],
    });
    if (typeof destinationPath !== "string") return;

    setError("");
    try {
      await invoke("save_graph_image", {
        sourcePath: result.outputPath,
        destinationPath,
      });
    } catch (reason) {
      setError(String(reason));
    }
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">C</span>ConTextation</div>
      </header>

      <section className="hero">
        <p className="eyebrow">TEXT TO GRAPH</p>
        <h1>글을 읽는<br /><em>새로운 방식</em></h1>
        <p className="intro">텍스트를 그래프로 시각화합니다.</p>
      </section>

      <section className="workspace">
        <aside className="control-panel">
          <div className="section-heading">
            <span>01</span><h2>분석할 글</h2>
          </div>
          <button className="file-picker" type="button" onClick={chooseFile}>
            <span className="file-icon">↗</span>
            <span>{inputPath ? fileName(inputPath) : "파일 선택"}</span>
            <small>{inputPath ? inputPath : "TXT · MD · PDF"}</small>
          </button>

          <div className="section-heading settings-heading">
            <span>02</span><h2>그래프 설정</h2>
          </div>
          <label>
            <span>최소 단어 빈도 <b>{minTermFreq}회</b></span>
            <input type="range" min="1" max="8" value={minTermFreq}
              onChange={(event) => setMinTermFreq(Number(event.target.value))} />
          </label>
          <label>
            <span>연결 강도 <b>{edgeThreshold.toFixed(2)}</b></span>
            <input type="range" min="0" max="0.2" step="0.01" value={edgeThreshold}
              onChange={(event) => setEdgeThreshold(Number(event.target.value))} />
          </label>
          <label>
            <span>최소 연결 수 <b>{minDegree}</b></span>
            <input type="range" min="0" max="4" value={minDegree}
              onChange={(event) => setMinDegree(Number(event.target.value))} />
          </label>

          <button className="generate-button" type="button" onClick={generateGraph}
            disabled={!inputPath || loading}>
            {loading ? "그래프를 만드는 중…" : "그래프 만들기"}
            <span>→</span>
          </button>
          <div className="graph-tips">
            <p><b>설정 팁</b> 그래프가 복잡하면 값을 올리고, 비어 있으면 낮춰 보세요.</p>
            <ul>
              <li><b>빈도</b> 단어가 등장한 횟수</li>
              <li><b>강도</b> 단어의 관계 강도</li>
              <li><b>연결 수</b> 다른 단어와 연결된 갯수</li>
            </ul>
          </div>
          {error && <p className="error-message">{error}</p>}
        </aside>

        <section className="canvas-panel" aria-live="polite">
          {result ? (
            <>
              <div className="canvas-topline">
                <div><p className="eyebrow">RESULT</p><h2>{fileName(inputPath)}</h2></div>
                <div className="result-actions">
                  <span className="ready-label">생성 완료</span>
                  <button className="save-button" type="button" onClick={saveGraphImage}>
                    PNG 저장
                  </button>
                </div>
              </div>
              <div className="graph-legend" aria-label="그래프 읽는 법">
                <span><i className="legend-node small" />원 크기 · 중심성</span>
                <span><i className="legend-line" />선 굵기 · 연결 강도</span>
                <span><i className="legend-node dark" />색 진함 · 영향도</span>
              </div>
              <img className="graph-image" src={result.imageDataUrl} alt={`${fileName(inputPath)} 지식 그래프`} />
              <p className="output-path">저장 위치 · {result.outputPath}</p>
            </>
          ) : (
            <div className="empty-canvas">
              <div className="orbit orbit-one" /><div className="orbit orbit-two" />
              <div className="empty-node central">주제</div>
              <div className="empty-node n1">쟁점</div><div className="empty-node n2">관계</div>
              <div className="empty-node n3">맥락</div>
              <div className="empty-guide" aria-label="그래프 사용 순서">
                <span><b>01</b> 파일 선택</span><i>→</i>
                <span><b>02</b> 설정</span><i>→</i>
                <span><b>03</b> 그래프 생성</span>
              </div>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}
