import type { AblationRow } from "../types";

interface AblationTableProps {
  rows: AblationRow[];
  k: number;
}

const format = (value: number) => value.toFixed(3);

/** The retrieval ablation as a table: each pipeline stage against its metrics.
 *
 * The last stage is highlighted as the one actually served, so a reader sees the
 * lift the reranker adds over the dense and hybrid baselines.
 */
export function AblationTable({ rows, k }: AblationTableProps) {
  return (
    <div className="table-wrap">
      <table className="metric-table">
        <thead>
          <tr>
            <th scope="col">Etapa</th>
            <th scope="col">Recall@{k}</th>
            <th scope="col">MRR</th>
            <th scope="col">nDCG@{k}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={row.name} className={i === rows.length - 1 ? "served" : undefined}>
              <th scope="row">{row.name}</th>
              <td>{format(row.recall_at_k)}</td>
              <td>{format(row.mrr)}</td>
              <td>{format(row.ndcg_at_k)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
