#!/usr/bin/env node
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const safeRealpath = (p) => { try { return fs.realpathSync(p); } catch { return ""; } };
const SELF_PATH = safeRealpath(fileURLToPath(import.meta.url));
const SELF_DIR = path.dirname(SELF_PATH);

export function isUsableBinary(candidate, platform = process.platform) {
  try {
    const stat = fs.statSync(candidate);
    if (!stat.isFile()) return false;
    if (platform !== "win32") {
      if ((stat.mode & 0o111) === 0) return false;
      fs.accessSync(candidate, fs.constants.X_OK);
    }
    const real = safeRealpath(candidate);
    if (!real || real === SELF_PATH || path.dirname(real) === SELF_DIR) return false;
    if (path.basename(path.dirname(candidate)) === ".bin" && stat.size < 4096) {
      if (fs.readFileSync(real, "utf8").includes("diodemcp.mjs")) return false;
    }
    return true;
  } catch {
    return false;
  }
}

export function findExecutable(name, env = process.env, platform = env.DIODEMCP_TEST_PLATFORM || process.platform) {
  const rawPath = env.PATH || "";
  const delim = platform === "win32" && rawPath.includes(";") ? ";" : path.delimiter;
  const exts = platform === "win32"
    ? [...new Set((env.PATHEXT || ".COM;.EXE;.BAT;.CMD").split(";").filter(Boolean).flatMap((e) => [e, e.toLowerCase()])), ""]
    : [""];
  for (const dir of rawPath.split(delim).filter(Boolean)) {
    for (const ext of exts) {
      const candidate = path.join(dir, `${name}${ext}`);
      if (isUsableBinary(candidate, platform)) return candidate;
    }
  }
  return null;
}

export function resolveLaunchTarget(args = process.argv.slice(2), env = process.env, platform = env.DIODEMCP_TEST_PLATFORM || process.platform) {
  if (env.DIODEMCP_BIN && isUsableBinary(env.DIODEMCP_BIN, platform)) return { command: env.DIODEMCP_BIN, args };
  const direct = findExecutable("diodemcp", env, platform);
  if (direct) return { command: direct, args };
  const uvx = findExecutable("uvx", env, platform);
  if (uvx) return { command: uvx, args: ["--quiet", "diodemcp", ...args] };
  const pipx = findExecutable("pipx", env, platform);
  if (pipx) return { command: pipx, args: ["run", "--quiet", "diodemcp", ...args] };
  return null;
}

if (process.argv[1] && safeRealpath(process.argv[1]) === SELF_PATH) {
  const target = resolveLaunchTarget();
  if (!target) {
    process.stderr.write(
      "[diodemcp] Error: Could not find 'diodemcp', 'uvx', or 'pipx' on PATH.\n" +
      "Install DiodeMCP with uv (recommended):\n" +
      "  curl -LsSf https://astral.sh/uv/install.sh | sh && uv tool install diodemcp\n" +
      "Or with pipx:\n  pipx install diodemcp\n"
    );
    process.exit(127);
  }
  const isWinCmd = process.platform === "win32" && /\.(cmd|bat)$/i.test(target.command);
  const child = spawn(target.command, target.args, { stdio: "inherit", shell: isWinCmd, windowsHide: true });
  for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) {
    process.on(sig, () => {
      if (!child.pid || child.exitCode !== null || child.signalCode !== null) return;
      if (process.platform === "win32") spawnSync("taskkill", ["/PID", String(child.pid), "/T", "/F"], { stdio: "ignore", windowsHide: true });
      try { child.kill(sig); } catch {}
    });
  }
  child.on("error", (err) => {
    process.stderr.write(`[diodemcp] Failed to launch '${target.command}': ${err.message}\n`);
    process.exit(127);
  });
  child.on("exit", (code, sig) => process.exit(sig ? 128 + (os.constants.signals[sig] || 1) : (code ?? 1)));
}
