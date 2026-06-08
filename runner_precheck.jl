using YAML
using PackageScanner

# Read configuration
vars = YAML.load_file(joinpath(ENV["GITHUB_WORKSPACE"], "_variables.yml"))
@info "Configuration loaded" vars

dest_path = joinpath(ENV["GITHUB_WORKSPACE"], "replication-package")

# ── Remote path: download via public Dropbox link ─────────────────────
url = get(ENV, "DROPBOX_DOWNLOAD_URL", nothing)

downloaded_ok = if !isnothing(url)
    @info "Downloading package from secret Dropbox link..."
    t0 = time()
    try
        run(`curl -fsSL -o package.zip $url`)
        @info "Download complete in $(round(time()-t0, digits=1))s"
        true
    catch e
        @error "curl download failed, will try local fallback" exception=e
        false
    end
else
    @info "No remote link configured — using local Dropbox path"
    false
end

# ── Local fallback: copy from Dropbox Apps folder ─────────────────────
if !downloaded_ok
    source_path = joinpath(ENV["JPE_DBOX_APPS"], vars["dropbox_rel_path"], "replication-package")
    @info "Copying package from local Dropbox..." source_path
    if !isdir(source_path)
        error("Package not found at $source_path")
    end
    isdir(dest_path) && rm(dest_path; recursive=true, force=true)
    PackageScanner.mycp(source_path, dest_path; recursive=true, force=true)
    @info "✓ Package copied from local Dropbox"
end

# ── Unzip downloaded archive (remote path only) ───────────────────────
if downloaded_ok && isfile("package.zip")
    @info "Unzipping Dropbox folder archive..."
    tmp_dir = mktempdir()
    try
        run(`unzip -oq package.zip -d $tmp_dir`)
    catch e
        @warn "unzip exited non-zero (may still be okay)" exception=e
    end
    rm("package.zip"; force=true)

    # Find all ZIPs inside the Dropbox folder
    candidates = filter(readdir(tmp_dir; join=true)) do f
        isfile(f) && endswith(lowercase(f), ".zip")
    end

    if isempty(candidates)
        error("No ZIP file found inside Dropbox folder archive")
    end

    @info "Found $(length(candidates)) ZIP(s) to extract" candidates
    isdir(dest_path) && rm(dest_path; recursive=true, force=true)
    mkpath(dest_path)

    for pkg_zip in candidates
        @info "Unzipping $pkg_zip..."
        try
            run(`unzip -oq $pkg_zip -d $dest_path`)
            if isdir(dest_path)
                rm_git(dest_path)
            end
        catch e
            @warn "unzip of $pkg_zip exited non-zero" exception=e
        end
    end

    rm(tmp_dir; recursive=true, force=true)
    @info "✓ Unzip complete"
end

if !isdir(dest_path)
    error("Package directory not found at $dest_path after download/copy step")
end

# ── PackageScanner precheck ────────────────────────────────────────────
pkg_size     = vars["package_size_gb"]
max_pkg_size = vars["package_max_pkg_size_gb"]
max_file_size = vars["package_max_file_size_gb"]

if pkg_size > max_pkg_size
    @info "Package >$(max_pkg_size) GB — using partial extraction mode"
    pkg_dir, manifest = PackageScanner.prepare_package_for_precheck(
        dest_path, size_threshold_gb=max_file_size, interactive=false)
    PackageScanner.precheck_package(pkg_dir, pre_manifest=manifest,
                                    no_data_scan=["__MACOSX", "renv"])
else
    @info "Unzipping files in $dest_path"
    try
        zips = PackageScanner.read_and_unzip_directory(dest_path)
        @info "Unzipped $(length(zips)) file(s)"
    catch e
        @warn "Unzip had issues (may be okay)" exception=e
    end
    @info "Running precheck on $dest_path"
    PackageScanner.precheck_package(dest_path, no_data_scan=["__MACOSX", "renv"])
    @info "✓ Precheck complete"
end
