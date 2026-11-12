import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const TEST_DIR = path.dirname(fileURLToPath(import.meta.url));
const PKG_ROOT = path.resolve(TEST_DIR, "..");
const BIN_PATH = path.join(PKG_ROOT, "bin", "diodemcp.mjs");
const NODE_BIN = process.execPath;
const NODE_DIR = path.dirname(NODE_BIN);

function makeTempDir(prefix = "diodemcp-npm-test-") {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

function writeExecutableScript(filePath, jsBody) {
  const script = `#!${NODE_BIN}\n${jsBody}\n`;
  fs.writeFileSync(filePath, script, { encoding: "utf8", mode: 0o755 });
  fs.chmodSync(filePath, 0o755);
  return filePath;
}

test("1. clean byte-for-byte stdio JSON-RPC pass-through with diodemcp on PATH and DIODEMCP_BIN override", (t) => {
  const tmp = makeTempDir();
  t.after(() => fs.rmSync(tmp, { recursive: true, force: true }));

  const mockBinDir = path.join(tmp, "bin");
  fs.mkdirSync(mockBinDir, { recursive: true });

  writeExecutableScript(
    path.join(mockBinDir, "diodemcp"),
    `
    const chunks = [];
    process.stdin.on("data", (c) => chunks.push(c));
    process.stdin.on("end", () => {
      const inputBuf = Buffer.concat(chunks);
      process.stderr.write("[mock-diodemcp-stderr] argv=" + JSON.stringify(process.argv.slice(2)) + "\\n");
      process.stdout.write(inputBuf);
    });
    `
  );

  const jsonRpcPayload = Buffer.from(
    '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}\n' +
      '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"read_file","arguments":{"path":".env"}}}\n',
    "utf8"
  );

  const res = spawnSync(
    NODE_BIN,
    [BIN_PATH, "shim", "--server", "filesystem", "--", "mcp-server-filesystem"],
    {
      input: jsonRpcPayload,
      env: {
        ...process.env,
        PATH: `${mockBinDir}${path.delimiter}${NODE_DIR}`,
        DIODEMCP_BIN: "",
      },
    }
  );

  assert.equal(res.status, 0);
  assert.equal(Buffer.compare(res.stdout, jsonRpcPayload), 0, "stdout must match JSON-RPC stream byte-for-byte with zero banner noise");
  assert.match(
    res.stderr.toString("utf8"),
    /\[mock-diodemcp-stderr\] argv=\["shim","--server","filesystem","--","mcp-server-filesystem"\]/
  );

  // Also verify DIODEMCP_BIN explicit override
  const overrideBin = writeExecutableScript(
    path.join(tmp, "custom-diodemcp"),
    `
    process.stderr.write("custom-override-used\\n");
    process.stdout.write(${JSON.stringify('{"jsonrpc":"2.0","id":99,"result":{"ok":true}}\n')});
    `
  );

  const resOverride = spawnSync(NODE_BIN, [BIN_PATH, "status", "--json"], {
    env: {
      ...process.env,
      PATH: mockBinDir,
      DIODEMCP_BIN: overrideBin,
    },
  });
  assert.equal(resOverride.status, 0);
  assert.equal(resOverride.stdout.toString("utf8"), '{"jsonrpc":"2.0","id":99,"result":{"ok":true}}\n');
  assert.match(resOverride.stderr.toString("utf8"), /custom-override-used/);
});

test("2. fallback resolution order: uvx (--quiet) -> pipx (run --quiet) -> stderr install hint (exit 127, empty stdout)", (t) => {
  const tmp = makeTempDir();
  t.after(() => fs.rmSync(tmp, { recursive: true, force: true }));

  const uvxDir = path.join(tmp, "uvx-only");
  const pipxDir = path.join(tmp, "pipx-only");
  const emptyDir = path.join(tmp, "empty");
  fs.mkdirSync(uvxDir, { recursive: true });
  fs.mkdirSync(pipxDir, { recursive: true });
  fs.mkdirSync(emptyDir, { recursive: true });

  // Mock uvx that verifies --quiet diodemcp is passed and echoes stdin to stdout
  writeExecutableScript(
    path.join(uvxDir, "uvx"),
    `
    process.stderr.write("[uvx-stderr] " + JSON.stringify(process.argv.slice(2)) + "\\n");
    process.stdin.pipe(process.stdout);
    `
  );

  // Mock pipx that verifies run --quiet diodemcp is passed
  writeExecutableScript(
    path.join(pipxDir, "pipx"),
    `
    process.stderr.write("[pipx-stderr] " + JSON.stringify(process.argv.slice(2)) + "\\n");
    process.stdin.pipe(process.stdout);
    `
  );

  const rpcFrame = '{"jsonrpc":"2.0","id":7,"result":{}}\n';

  // (a) Both uvxDir and pipxDir on PATH -> must prefer uvx with --quiet
  const uvxRes = spawnSync(NODE_BIN, [BIN_PATH, "init", "--tier0-only"], {
    input: rpcFrame,
    encoding: "utf8",
    env: {
      ...process.env,
      PATH: `${uvxDir}${path.delimiter}${pipxDir}`,
      DIODEMCP_BIN: "",
    },
  });
  assert.equal(uvxRes.status, 0);
  assert.equal(uvxRes.stdout, rpcFrame, "uvx fallback must not pollute stdout");
  assert.match(uvxRes.stderr, /\[uvx-stderr\] \["--quiet","diodemcp","init","--tier0-only"\]/);

  // (b) Only pipxDir on PATH -> must fall back to pipx run --quiet diodemcp
  const pipxRes = spawnSync(NODE_BIN, [BIN_PATH, "doctor"], {
    input: rpcFrame,
    encoding: "utf8",
    env: {
      ...process.env,
      PATH: pipxDir,
      DIODEMCP_BIN: "",
    },
  });
  assert.equal(pipxRes.status, 0);
  assert.equal(pipxRes.stdout, rpcFrame, "pipx fallback must not pollute stdout");
  assert.match(pipxRes.stderr, /\[pipx-stderr\] \["run","--quiet","diodemcp","doctor"\]/);

  // (c) Neither diodemcp, uvx, nor pipx on PATH -> exit 127 (or 2), zero stdout, helpful stderr install hint
  const missingRes = spawnSync(NODE_BIN, [BIN_PATH, "init"], {
    encoding: "utf8",
    env: {
      ...process.env,
      PATH: emptyDir,
      DIODEMCP_BIN: "",
    },
  });
  assert.ok(
    missingRes.status === 127 || missingRes.status === 2,
    `Expected exit status 127 or 2 when no runtime found, got ${missingRes.status}`
  );
  assert.equal(missingRes.stdout, "", "stdout must remain 100% empty on missing binary error");
  assert.match(missingRes.stderr, /uv tool install diodemcp|astral\.sh\/uv/i);
  assert.match(missingRes.stderr, /pipx/i);
});

test("3. self-recursion guard skips symlinked node_modules/.bin/diodemcp and non-executable files on PATH", (t) => {
  const tmp = makeTempDir();
  t.after(() => fs.rmSync(tmp, { recursive: true, force: true }));

  const nodeModulesBin = path.join(tmp, "node_modules", ".bin");
  const nonExecDir = path.join(tmp, "non-exec");
  const realBinDir = path.join(tmp, "real-bin");
  fs.mkdirSync(nodeModulesBin, { recursive: true });
  fs.mkdirSync(nonExecDir, { recursive: true });
  fs.mkdirSync(realBinDir, { recursive: true });

  // 1) Symlink node_modules/.bin/diodemcp -> packages/npm-diodemcp/bin/diodemcp.mjs
  const symlinkPath = path.join(nodeModulesBin, "diodemcp");
  fs.symlinkSync(BIN_PATH, symlinkPath);

  // 2) Non-executable diodemcp file (mode 0644) ahead of real binary
  const nonExecFile = path.join(nonExecDir, "diodemcp");
  fs.writeFileSync(nonExecFile, "#!/bin/sh\necho SHOULD_NOT_RUN\n", { mode: 0o644 });
  fs.chmodSync(nonExecFile, 0o644);

  // 3) Real mock diodemcp binary after both in PATH
  writeExecutableScript(
    path.join(realBinDir, "diodemcp"),
    `
    process.stdout.write("RESOLVED_REAL_DIODEMCP:" + process.argv.slice(2).join(",") + "\\n");
    `
  );

  // Invoke via the symlink itself with node_modules/.bin, bin/, and nonExecDir all ahead of realBinDir!
  const pathEnv = [
    nodeModulesBin,
    path.dirname(BIN_PATH),
    nonExecDir,
    realBinDir,
    NODE_DIR,
  ].join(path.delimiter);

  const res = spawnSync(NODE_BIN, [symlinkPath, "audit", "--limit", "5"], {
    encoding: "utf8",
    env: {
      ...process.env,
      PATH: pathEnv,
      DIODEMCP_BIN: symlinkPath, // Even if DIODEMCP_BIN points to self symlink, must not recurse!
    },
  });

  assert.equal(res.status, 0);
  assert.equal(res.stdout, "RESOLVED_REAL_DIODEMCP:audit,--limit,5\n");

  // 4) Also verify symlink first on PATH falls back cleanly to uvx when no real diodemcp binary exists
  const uvxOnlyDir = path.join(tmp, "uvx-fallback");
  fs.mkdirSync(uvxOnlyDir, { recursive: true });
  writeExecutableScript(
    path.join(uvxOnlyDir, "uvx"),
    `
    process.stdout.write("UVX_AFTER_SYMLINK:" + process.argv.slice(2).join(",") + "\\n");
    `
  );
  const resUvxAfterSymlink = spawnSync(NODE_BIN, [symlinkPath, "status"], {
    encoding: "utf8",
    env: {
      ...process.env,
      PATH: [nodeModulesBin, path.dirname(BIN_PATH), nonExecDir, uvxOnlyDir, NODE_DIR].join(path.delimiter),
      DIODEMCP_BIN: "",
    },
  });
  assert.equal(resUvxAfterSymlink.status, 0);
  assert.equal(resUvxAfterSymlink.stdout, "UVX_AFTER_SYMLINK:--quiet,diodemcp,status\n");
});

test("4. cross-platform win32 PATHEXT resolution", async (t) => {
  const tmp = makeTempDir();
  t.after(() => fs.rmSync(tmp, { recursive: true, force: true }));

  const winDir1 = path.join(tmp, "win-self");
  const winDir2 = path.join(tmp, "win-uvx");
  fs.mkdirSync(winDir1, { recursive: true });
  fs.mkdirSync(winDir2, { recursive: true });

  // Self symlink as diodemcp.cmd in winDir1
  fs.symlinkSync(BIN_PATH, path.join(winDir1, "diodemcp.cmd"));

  // Mock uvx.exe in winDir2
  const uvxExe = path.join(winDir2, "uvx.exe");
  fs.writeFileSync(uvxExe, "MZ-mock", { mode: 0o644 });

  const mod = await import("../bin/diodemcp.mjs");
  const resolved = mod.resolveLaunchTarget(
    ["shim", "--server", "s1"],
    {
      PATH: `${winDir1};${winDir2}`,
      PATHEXT: ".COM;.EXE;.BAT;.CMD",
    },
    "win32"
  );

  assert.ok(resolved, "should resolve uvx.exe on win32 via PATHEXT");
  assert.equal(resolved.command, uvxExe);
  assert.deepEqual(resolved.args, ["--quiet", "diodemcp", "shim", "--server", "s1"]);
});

test("5. exact exit-code propagation (0, 2, 42) and SIGTERM/SIGINT signal forwarding", async (t) => {
  const tmp = makeTempDir();
  t.after(() => fs.rmSync(tmp, { recursive: true, force: true }));

  const binDir = path.join(tmp, "bin");
  fs.mkdirSync(binDir, { recursive: true });

  writeExecutableScript(
    path.join(binDir, "diodemcp"),
    `
    const mode = process.argv[2];
    if (mode === "exit-code") {
      process.exit(Number(process.argv[3]));
    }
    if (mode === "trap-signal") {
      const targetSig = process.argv[3] || "SIGTERM";
      process.on(targetSig, () => {
        process.stderr.write("CHILD_CAUGHT_" + targetSig + "\\n");
        process.exit(targetSig === "SIGINT" ? 130 : 143);
      });
      process.stdout.write("READY\\n");
      setInterval(() => {}, 1000);
    }
    `
  );

  const env = {
    ...process.env,
    PATH: `${binDir}${path.delimiter}${NODE_DIR}`,
    DIODEMCP_BIN: "",
  };

  for (const expectedCode of [0, 2, 42]) {
    const r = spawnSync(NODE_BIN, [BIN_PATH, "exit-code", String(expectedCode)], { env });
    assert.equal(r.status, expectedCode, `exit code ${expectedCode} must propagate exactly`);
  }

  for (const [sig, expectedExit] of [
    ["SIGTERM", 143],
    ["SIGINT", 130],
  ]) {
    const child = spawn(NODE_BIN, [BIN_PATH, "trap-signal", sig], {
      env,
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stderrOut = "";
    child.stderr.on("data", (d) => {
      stderrOut += d.toString("utf8");
    });

    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`Timed out waiting for READY (${sig})`)), 5000);
      child.stdout.on("data", (d) => {
        if (d.toString("utf8").includes("READY")) {
          clearTimeout(timer);
          resolve();
        }
      });
      child.on("error", reject);
    });

    child.kill(sig);

    const exitCode = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`Timed out waiting for exit (${sig})`)), 5000);
      child.on("close", (code) => {
        clearTimeout(timer);
        resolve(code);
      });
    });

    assert.equal(exitCode, expectedExit, `signal ${sig} must propagate child exit code ${expectedExit}`);
    assert.match(stderrOut, new RegExp(`CHILD_CAUGHT_${sig}`));
  }
});

test("6. real integration smoke test with packages/diodemcp/.venv/bin on PATH", (t) => {
  const venvBin = path.resolve(PKG_ROOT, "..", "diodemcp", ".venv", "bin");
  const realDiodemcp = path.join(venvBin, "diodemcp");
  if (!fs.existsSync(realDiodemcp)) {
    t.skip("packages/diodemcp/.venv/bin/diodemcp not present");
    return;
  }

  const env = {
    ...process.env,
    PATH: `${venvBin}${path.delimiter}${process.env.PATH || ""}`,
    DIODEMCP_BIN: "",
  };

  // (a) --help works via launcher
  const helpRes = spawnSync(NODE_BIN, [BIN_PATH, "--help"], {
    encoding: "utf8",
    env,
  });
  assert.equal(helpRes.status, 0);
  assert.match(helpRes.stdout, /Usage:\s+diodemcp/i);

  // (b) fail-closed hook invocation via launcher preserves exact JSON stdout & exit code 2
  const hookRes = spawnSync(
    NODE_BIN,
    [BIN_PATH, "hook", "claude", "PreToolUse"],
    {
      input: "{corrupt-json-stdin",
      encoding: "utf8",
      env,
    }
  );
  assert.equal(hookRes.status, 2);
  const parsed = JSON.parse(hookRes.stdout.trim());
  assert.equal(parsed.hookSpecificOutput.permissionDecision, "deny");
});

