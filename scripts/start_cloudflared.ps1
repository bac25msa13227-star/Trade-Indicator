$null = docker rm -f xauusd-cloudflared 2>&1
Write-Host "Old container removed (if any)"

$target = "http://xauusd-nginx:80"
$proto = "http2"
$args_arr = @("tunnel", "--no-autoupdate", "--protocol", $proto, "--url", $target)

$cid = docker run -d --name xauusd-cloudflared --network trade-indicator_default --restart unless-stopped cloudflare/cloudflared:latest @args_arr
Write-Host "Container ID: $cid"
Write-Host "Exit code: $LASTEXITCODE"

Start-Sleep 5
Write-Host "`n--- Containers ---"
docker ps -a --filter "name=cloudflared"

Write-Host "`n--- Logs ---"
docker logs xauusd-cloudflared --tail 20 2>&1
    