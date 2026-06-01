import { useAppStore } from "@/store/appStore";
import { cn } from "@/lib/utils";

export default function ModeToggle() {
  const mode = useAppStore((s) => s.mode);
  const toggleMode = useAppStore((s) => s.toggleMode);

  return (
    <button
      onClick={toggleMode}
      className={cn(
        "flex items-center gap-1.5 rounded-m3-full px-m3-3 py-m3-1.5 text-label-md font-medium transition-colors",
        "border border-m3-outline-variant hover:bg-m3-surface-container-high"
      )}
      title={mode === "simple" ? "切换到专业模式" : "切换到简约模式"}
    >
      <span
        className={cn(
          "h-2 w-2 rounded-full transition-colors",
          mode === "simple" ? "bg-m3-primary" : "bg-m3-tertiary"
        )}
      />
      <span className="text-m3-on-surface-variant">
        {mode === "simple" ? "简约" : "专业"}
      </span>
    </button>
  );
}
