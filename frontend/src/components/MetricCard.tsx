interface MetricCardProps {
  label: string;
  value: string;
  detail?: string;
}

/** A single headline statistic: a big value with a label and optional detail. */
export function MetricCard({ label, value, detail }: MetricCardProps) {
  return (
    <div className="metric-card">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
      {detail && <span className="metric-detail">{detail}</span>}
    </div>
  );
}
