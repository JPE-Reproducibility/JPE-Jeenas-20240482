using YAML
using PackageScanner

# Read configuration
vars = YAML.load_file(joinpath(ENV["GITHUB_WORKSPACE"], "_variables.yml"))

@info "Configuration loaded" vars

# Construct paths
dropbox_base = get(ENV, "DROPBOX_BASE", joinpath(ENV["HOME"], "Dropbox"))
source_path = joinpath(dropbox_base, vars["dropbox_rel_path"], "replication-package")
dest_path = joinpath(ENV["GITHUB_WORKSPACE"], "replication-package")

# Copy package
@info "Copying package" source_path dest_path
if isdir(source_path)
    cp(source_path, dest_path; force=true)
    @info "✓ Package copied successfully"
else
    error("✗ Package not found at \$source_path")
end

# Unzip files
@info "Unzipping files in \$dest_path"
zips = PackageScanner.read_and_unzip_directory(dest_path)
@info "Unzipped" zips

# Run precheck
@info "Running precheck on \$dest_path"
PackageScanner.precheck_package(dest_path)
        
@info "✓ Precheck complete"
