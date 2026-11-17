// Small shadcn-style primitives (hand-written so the bundle stays tiny).
import { clsx, type ClassValue } from 'clsx';
import { forwardRef, type ButtonHTMLAttributes, type HTMLAttributes, type InputHTMLAttributes, type ReactNode } from 'react';
import { twMerge } from 'tailwind-merge';

export const cn = (...inputs: ClassValue[]) => twMerge(clsx(inputs));

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'outline' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
};
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'outline', size = 'md', type = 'button', ...props }, ref) => (
    <button
      ref={ref}
      type={type}
      className={cn(
        'inline-flex select-none items-center justify-center gap-2 whitespace-nowrap rounded font-medium transition-colors',
        'disabled:pointer-events-none disabled:opacity-45',
        size === 'sm' && 'h-8 px-3 text-[13px]',
        size === 'md' && 'h-10 px-4 text-sm',
        size === 'lg' && 'h-12 px-6 text-base',
        variant === 'primary' && 'bg-hivis text-[#0b0d10] hover:brightness-110 active:brightness-95',
        variant === 'outline' && 'border border-line bg-surface text-ink hover:bg-raised',
        variant === 'ghost' && 'text-ink hover:bg-raised',
        variant === 'danger' && 'border border-danger/50 text-danger hover:bg-danger/10',
        className,
      )}
      {...props}
    />
  ),
);
Button.displayName = 'Button';

export function Panel({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('panel', className)} {...props} />;
}

export function Eyebrow({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('eyebrow', className)} {...props} />;
}

export function Switch({
  checked, onChange, label, hint, id,
}: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string; id: string }) {
  return (
    <label htmlFor={id} className="flex cursor-pointer items-start justify-between gap-4 py-2">
      <span>
        <span className="block text-sm">{label}</span>
        {hint && <span className="block text-xs text-muted">{hint}</span>}
      </span>
      <button
        id={id}
        role="switch"
        type="button"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cn(
          'relative mt-0.5 h-6 w-11 shrink-0 rounded-full border transition-colors',
          checked ? 'border-hivis bg-hivis' : 'border-line bg-raised',
        )}
      >
        <span
          className={cn(
            'absolute top-0.5 h-[18px] w-[18px] rounded-full transition-all',
            checked ? 'left-[22px] bg-[#0b0d10]' : 'left-0.5 bg-muted',
          )}
        />
      </button>
    </label>
  );
}

export function Range({
  label, value, display, className, ...props
}: InputHTMLAttributes<HTMLInputElement> & { label: string; display: string }) {
  return (
    <label className={cn('block py-2', className)}>
      <span className="flex items-baseline justify-between text-sm">
        <span>{label}</span>
        <span className="num text-xs text-muted">{display}</span>
      </span>
      <input type="range" value={value} className="mt-2 w-full accent-[rgb(var(--hivis))]" {...props} />
    </label>
  );
}

export function Segmented<T extends string>({
  value, onChange, options, label,
}: { value: T; onChange: (v: T) => void; options: { value: T; label: ReactNode; title?: string }[]; label: string }) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded border border-line bg-raised p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          type="button"
          title={o.title}
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={cn(
            'rounded-[4px] px-3 py-1.5 text-[13px] font-medium transition-colors',
            value === o.value ? 'bg-surface text-ink shadow-sm' : 'text-muted hover:text-ink',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Chip({ className, ...props }: HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={cn('inline-flex items-center gap-1 rounded-full border border-line bg-raised px-2.5 py-0.5 text-xs text-muted', className)}
      {...props}
    />
  );
}

export function PageHeader({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) {
  return (
    <header className="mb-6">
      <Eyebrow>{eyebrow}</Eyebrow>
      <h1 className="mt-1 text-2xl font-semibold sm:text-3xl">{title}</h1>
      {children && <p className="mt-2 max-w-2xl text-sm text-muted">{children}</p>}
    </header>
  );
}
