-- lard_bridge.lua — FlyWithLua bridge for LARD-LAAS-TAF
-- =====================================================
-- Reads command.json from exchange dir, applies datarefs natively
-- in the render loop, writes status.json with ack + actual pose.
--
-- Installation:
--   Copy to X-Plane 12/Resources/plugins/FlyWithLua/Scripts/
--
-- Protocol:
--   Python writes command.json {seq, action, drefs, weather}
--   Lua reads it every frame, applies datarefs, writes status.json
--   Python polls status.json for ack_seq == seq

-- =========================================================================
-- Configuration
-- =========================================================================

-- Exchange directory: use FlyWithLua's SCRIPT_DIRECTORY constant
local SEP = package.config:sub(1, 1)  -- "/" on Linux, "\" on Windows
local EXCHANGE_DIR = SCRIPT_DIRECTORY .. "lard_exchange" .. SEP

local CMD_FILE = EXCHANGE_DIR .. "command.json"
local STS_FILE = EXCHANGE_DIR .. "status.json"
local STS_TMP  = EXCHANGE_DIR .. "status.tmp"

local last_ack_seq = -1

-- =========================================================================
-- Minimal JSON parser (handles nested objects one level deep)
-- =========================================================================

local function json_parse(str)
    if not str then return nil end
    local obj = {}
    str = str:match("^%s*{(.+)}%s*$")
    if not str then return nil end

    local pos = 1
    local len = #str

    local function skip_ws()
        while pos <= len and str:sub(pos, pos):match("%s") do pos = pos + 1 end
    end

    local function parse_string()
        skip_ws()
        if str:sub(pos, pos) ~= '"' then return nil end
        pos = pos + 1
        local start = pos
        while pos <= len and str:sub(pos, pos) ~= '"' do
            if str:sub(pos, pos) == '\\' then pos = pos + 1 end
            pos = pos + 1
        end
        local s = str:sub(start, pos - 1)
        pos = pos + 1
        return s
    end

    local function parse_value()
        skip_ws()
        local c = str:sub(pos, pos)
        if c == '"' then
            return parse_string()
        elseif c == '{' then
            local nested = {}
            pos = pos + 1
            skip_ws()
            if str:sub(pos, pos) == '}' then
                pos = pos + 1
                return nested
            end
            while pos <= len do
                skip_ws()
                local k = parse_string()
                if not k then break end
                skip_ws()
                if str:sub(pos, pos) == ':' then pos = pos + 1 end
                skip_ws()
                local v = parse_value()
                nested[k] = v
                skip_ws()
                if str:sub(pos, pos) == ',' then
                    pos = pos + 1
                elseif str:sub(pos, pos) == '}' then
                    pos = pos + 1
                    break
                end
            end
            return nested
        elseif c == 't' then
            pos = pos + 4; return true
        elseif c == 'f' then
            pos = pos + 5; return false
        elseif c == 'n' then
            pos = pos + 4; return nil
        else
            local start = pos
            if str:sub(pos, pos) == '-' then pos = pos + 1 end
            while pos <= len and str:sub(pos, pos):match("[%d%.eE%+%-]") do
                pos = pos + 1
            end
            return tonumber(str:sub(start, pos - 1))
        end
    end

    while pos <= len do
        skip_ws()
        if pos > len then break end
        local key = parse_string()
        if not key then break end
        skip_ws()
        if str:sub(pos, pos) == ':' then pos = pos + 1 end
        skip_ws()
        local val = parse_value()
        obj[key] = val
        skip_ws()
        if str:sub(pos, pos) == ',' then pos = pos + 1 end
    end

    return obj
end

-- =========================================================================
-- JSON serializer
-- =========================================================================

local function json_number(v)
    if v == nil then return "null" end
    return string.format("%.12g", v)
end

local function json_serialize(obj, indent)
    indent = indent or ""
    local inner = indent .. "  "
    local parts = {}
    for k, v in pairs(obj) do
        local val_str
        if type(v) == "table" then
            val_str = json_serialize(v, inner)
        elseif type(v) == "string" then
            val_str = '"' .. v .. '"'
        elseif type(v) == "boolean" then
            val_str = v and "true" or "false"
        elseif type(v) == "number" then
            val_str = json_number(v)
        else
            val_str = "null"
        end
        parts[#parts + 1] = inner .. '"' .. tostring(k) .. '": ' .. val_str
    end
    return "{\n" .. table.concat(parts, ",\n") .. "\n" .. indent .. "}"
end

-- =========================================================================
-- File I/O helpers
-- =========================================================================

local function read_file(path)
    local f = io.open(path, "r")
    if not f then return nil end
    local content = f:read("*a")
    f:close()
    return content
end

local function write_file_atomic(path, tmp_path, content)
    local f = io.open(tmp_path, "w")
    if not f then return false end
    f:write(content)
    f:close()
    os.remove(path)
    return os.rename(tmp_path, path)
end

-- =========================================================================
-- Dataref helpers — using FlyWithLua native set()/get()
-- =========================================================================

local has_xplm = (type(XPLMFindDataRef) == "function")

local dref_cache = {}

local function get_dref_handle(name)
    if not has_xplm then return nil end
    if dref_cache[name] then return dref_cache[name] end
    local ref = XPLMFindDataRef(name)
    if ref then dref_cache[name] = ref end
    return ref
end

local function smart_set(name, value)
    set(name, value)
end

local function smart_get(name)
    return get(name)
end

local function get_double(name)
    if has_xplm then
        local ref = get_dref_handle(name)
        if ref then return XPLMGetDatad(ref) end
    end
    return get(name)
end

-- =========================================================================
-- Read current pose from X-Plane
-- =========================================================================

local function read_current_pose()
    return {
        lat     = get_double("sim/flightmodel/position/latitude")  or 0,
        lon     = get_double("sim/flightmodel/position/longitude") or 0,
        alt_m   = smart_get("sim/flightmodel/position/elevation")  or 0,
        heading = smart_get("sim/flightmodel/position/psi")        or 0,
        pitch   = smart_get("sim/flightmodel/position/theta")      or 0,
        roll    = smart_get("sim/flightmodel/position/phi")        or 0,
    }
end

local function read_reference_point()
    return {
        lat     = get_double("sim/flightmodel/position/latitude")  or 0,
        lon     = get_double("sim/flightmodel/position/longitude") or 0,
        elev    = smart_get("sim/flightmodel/position/elevation")  or 0,
        local_x = get_double("sim/flightmodel/position/local_x")   or 0,
        local_y = get_double("sim/flightmodel/position/local_y")   or 0,
        local_z = get_double("sim/flightmodel/position/local_z")   or 0,
    }
end

local function read_pilot_eye()
    return {
        x = smart_get("sim/aircraft/view/acf_peX") or 0,
        y = smart_get("sim/aircraft/view/acf_peY") or 0,
        z = smart_get("sim/aircraft/view/acf_peZ") or 0,
    }
end

-- =========================================================================
-- Action handlers
-- =========================================================================

local function handle_setup(cmd)
    local status = {
        ok = true,
        ref_point = read_reference_point(),
        pilot_eye = read_pilot_eye(),
        fov_deg = smart_get("sim/graphics/view/field_of_view_deg") or 65.0,
        actual_pose = read_current_pose(),
    }
    return status
end

local function handle_set_pose(cmd)
    if cmd.drefs and type(cmd.drefs) == "table" then
        for name, value in pairs(cmd.drefs) do
            if type(value) == "number" then
                smart_set(name, value)
            end
        end
    end

    if cmd.weather and type(cmd.weather) == "table" then
        for name, value in pairs(cmd.weather) do
            if type(value) == "number" then
                smart_set(name, value)
            end
        end
    end

    return {ok = true, actual_pose = read_current_pose()}
end

local function handle_read_pose(cmd)
    return {
        ok = true,
        actual_pose = read_current_pose(),
        ref_point = read_reference_point(),
    }
end

local function handle_release(cmd)
    return {ok = true, actual_pose = read_current_pose()}
end

-- =========================================================================
-- Main tick — called every draw frame
-- =========================================================================

function lard_bridge_tick()
    local raw = read_file(CMD_FILE)
    if not raw then return end

    local ok_parse, cmd = pcall(json_parse, raw)
    if not ok_parse or not cmd or not cmd.seq then return end

    if cmd.seq == last_ack_seq then return end

    local status
    local action = cmd.action or "noop"

    local ok_exec, result = pcall(function()
        if action == "setup" then
            return handle_setup(cmd)
        elseif action == "set_pose" then
            return handle_set_pose(cmd)
        elseif action == "read_pose" then
            return handle_read_pose(cmd)
        elseif action == "release" then
            return handle_release(cmd)
        elseif action == "noop" then
            return {ok = true, actual_pose = read_current_pose()}
        else
            return {ok = false, error = "unknown action: " .. tostring(action)}
        end
    end)

    if ok_exec then
        status = result
    else
        status = {ok = false, error = tostring(result)}
    end

    status.ack_seq = cmd.seq
    status.timestamp = os.clock()
    local json_str = json_serialize(status)
    write_file_atomic(STS_FILE, STS_TMP, json_str)
    last_ack_seq = cmd.seq
end

-- =========================================================================
-- Register with FlyWithLua
-- =========================================================================

-- Ensure exchange directory exists
os.execute('mkdir "' .. EXCHANGE_DIR:gsub("/", "\\") .. '" 2>nul')

-- Clean stale files on load
os.remove(CMD_FILE)
os.remove(STS_FILE)
os.remove(STS_TMP)

do_every_draw("lard_bridge_tick()")

logMsg("LARD bridge loaded — exchange dir: " .. EXCHANGE_DIR)
