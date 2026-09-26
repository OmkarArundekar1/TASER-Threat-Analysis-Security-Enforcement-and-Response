/**
 * CampaignId — the single canonical way to render a campaign identifier
 * anywhere in the Overview or Incident GUIs. Always shows the raw,
 * untruncated campaign_id from the backend — never a campaign_label,
 * name, or other derived value. One place to change the visual style
 * (size/weight/color) instead of duplicating font-mono/color classes
 * at every call site.
 */

const SIZE_CLASSES = {
  sm: 'text-xs',
  md: 'text-sm',
  lg: 'text-lg',
} as const;

export function CampaignId({
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
      className={`font-mono font-bold text-orange-400 tracking-tight break-all ${SIZE_CLASSES[size]} ${className}`}
    >
      {id}
    </span>
  );
}
