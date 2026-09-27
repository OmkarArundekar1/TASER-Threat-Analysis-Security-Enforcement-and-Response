/**
 * OperationId — the single canonical way to render an operation
 * identifier anywhere in the GUI, mirroring CampaignId.tsx. Always
 * shows the raw, untruncated operation_id from the backend. Uses the
 * fuchsia color already established for operation_id elsewhere (see
 * IncidentView.tsx's IncidentHeader) so a reader learns one color means
 * "this is an operation" everywhere, not just in one place.
 */

const SIZE_CLASSES = {
  sm: 'text-xs',
  md: 'text-sm',
  lg: 'text-lg',
} as const;

export function OperationId({
  id,
  size = 'md',
  className = '',
}: {
  id: string;
  size?: keyof typeof SIZE_CLASSES;
  className?: string;
}) {
  return (
    <span
      className={`font-mono font-bold text-fuchsia-400 tracking-tight break-all ${SIZE_CLASSES[size]} ${className}`}
    >
      {id}
    </span>
  );
}
