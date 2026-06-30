import { quality } from "../quality";
import type { MetricCI, RetrievalStats } from "../types";
import { AblationTable } from "./AblationTable";
import { MetricCard } from "./MetricCard";

/** Format a 0–1 ratio as a whole-ish percentage (e.g. 0.913 → "91.3 %"). */
const pct = (value: number) => `${(value * 100).toFixed(1)} %`;

/** Render a confidence interval as a compact "IC 95% lo–hi %" detail string. */
function ciDetail(ci: MetricCI | null): string | undefined {
  if (!ci) {
    return undefined;
  }
  return `IC 95 % ${pct(ci.low)} – ${pct(ci.high)}`;
}

/** A retrieval eval set's three headline metrics as a row of cards. */
function RetrievalCards({ stats }: { stats: RetrievalStats }) {
  const recallDetail = ciDetail(stats.recall_ci);
  const mrrDetail = ciDetail(stats.mrr_ci);
  return (
    <div className="metric-grid">
      <MetricCard
        label={`Recall@${stats.k}`}
        value={pct(stats.recall_at_k)}
        {...(recallDetail ? { detail: recallDetail } : {})}
      />
      <MetricCard
        label="MRR"
        value={stats.mrr.toFixed(3)}
        {...(mrrDetail ? { detail: mrrDetail } : {})}
      />
      <MetricCard label={`nDCG@${stats.k}`} value={stats.ndcg_at_k.toFixed(3)} />
    </div>
  );
}

/** The quality page: headline eval metrics rendered from the committed reports. */
export function QualityPage() {
  const { retrieval, ablation, generation, security } = quality;
  const generatedDate = new Date(quality.generated).toLocaleDateString("es-ES", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  return (
    <div className="quality">
      <p className="quality-intro">
        Cada cifra procede de un <em>eval</em> reproducible versionado en el repositorio
        (<code>reports/</code>); no son estimaciones. La recuperación se mide con
        equivalencia textual; las respuestas, con un juez LLM.
      </p>

      <section className="quality-section">
        <h2>Recuperación</h2>
        <h3 className="quality-subhead">
          Conjunto oro · {retrieval.gold.num_queries} preguntas curadas a mano
        </h3>
        <RetrievalCards stats={retrieval.gold} />
        <h3 className="quality-subhead">
          Conjunto plata · {retrieval.silver.num_queries.toLocaleString("es-ES")}{" "}
          preguntas generadas automáticamente
        </h3>
        <RetrievalCards stats={retrieval.silver} />
      </section>

      <section className="quality-section">
        <h2>Pipeline de recuperación</h2>
        <p className="quality-note">
          Ablación sobre el conjunto oro: el <strong>cross-encoder</strong> reordena los
          candidatos del híbrido y es la etapa que se sirve en producción.
        </p>
        <AblationTable rows={ablation.rows} k={ablation.k} />
      </section>

      <section className="quality-section">
        <h2>Calidad de las respuestas</h2>
        <p className="quality-note">
          Juez LLM sobre {generation.num_queries} preguntas de extremo a extremo.
        </p>
        <div className="metric-grid">
          <MetricCard label="Fidelidad" value={pct(generation.mean_faithfulness)} />
          <MetricCard label="Corrección" value={pct(generation.mean_correctness)} />
          <MetricCard
            label="Tasa de rechazo"
            value={pct(generation.refusal_rate)}
            detail="rechaza cuando no hay base"
          />
        </div>
      </section>

      <section className="quality-section">
        <h2>Robustez adversarial</h2>
        <p className="quality-note">
          {security.num_passed} de {security.num_cases} casos superados (
          {pct(security.pass_rate)}).
        </p>
        <ul className="security-list">
          {security.by_category.map((cat) => (
            <li key={cat.name} className="security-row">
              <span className="security-name">{cat.name.replace(/_/g, " ")}</span>
              <span className="security-bar" aria-hidden="true">
                <span
                  className="security-bar-fill"
                  style={{ width: `${cat.pass_rate * 100}%` }}
                />
              </span>
              <span className="security-pct">{pct(cat.pass_rate)}</span>
            </li>
          ))}
        </ul>
      </section>

      <p className="quality-generated">Datos generados el {generatedDate}.</p>
    </div>
  );
}
