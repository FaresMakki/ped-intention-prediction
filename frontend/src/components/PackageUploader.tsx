import { useRef } from "react";

interface PackageUploaderProps {
  uploading: boolean;
  onUpload: (file: File) => Promise<void>;
}

export function PackageUploader({ uploading, onUpload }: PackageUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  async function chooseFile(file: File | undefined) {
    if (!file) return;
    try {
      await onUpload(file);
    } finally {
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <div className="package-uploader">
      <input
        accept=".zip,application/zip"
        aria-label="Choose a PIE package ZIP"
        disabled={uploading}
        onChange={(event) => chooseFile(event.target.files?.[0])}
        ref={inputRef}
        type="file"
      />
      <button
        disabled={uploading}
        onClick={() => inputRef.current?.click()}
        type="button"
      >
        <span className="upload-icon" aria-hidden="true">
          {uploading ? "···" : "+"}
        </span>
        <span>
          <strong>{uploading ? "Validating package" : "Upload package"}</strong>
          <small>Version 1 ZIP · maximum 25 MB</small>
        </span>
      </button>
    </div>
  );
}
