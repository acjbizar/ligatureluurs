<?php
/**
 * templates/element/character.php
 *
 * Usage:
 *   echo $this->element('character', ['ch' => 'A']);
 *   // or:
 *   echo $this->element('character', ['character' => 'A']);
 *
 * Expects SVGs in: ROOT/sketches/character-uXXXX.svg
 * Example: 'A' (U+0041) -> sketches/character-u0041.svg
 */

declare(strict_types=1);

// Accept either 'ch' or 'character'
$ch = $ch ?? $character ?? null;

// Optional behavior controls
$strict = $strict ?? false;            // if true, throw when missing/invalid
$fallback = $fallback ?? '';           // HTML/SVG to output when missing (if not strict)

// Basic validation
if (!is_string($ch) || $ch === '') {
    if ($strict) {
        throw new InvalidArgumentException('Element "character" requires a non-empty string $ch (or $character).');
    }
    echo $fallback;
    return;
}

// Take the first unicode character (in case a longer string is passed)
$ch = mb_substr($ch, 0, 1, 'UTF-8');

// Get Unicode code point (works on PHP 7.2+ with mbstring; intl is optional)
$codepoint = null;

if (class_exists(\IntlChar::class)) {
    $codepoint = \IntlChar::ord($ch);
} elseif (function_exists('mb_ord')) {
    $codepoint = mb_ord($ch, 'UTF-8');
} else {
    // Last-resort polyfill (should rarely be needed on modern PHP)
    $u = mb_convert_encoding($ch, 'UCS-4BE', 'UTF-8');
    $arr = unpack('N', $u);
    $codepoint = $arr[1] ?? null;
}

if (!is_int($codepoint) || $codepoint < 0) {
    if ($strict) {
        throw new RuntimeException('Could not determine Unicode code point for input character.');
    }
    echo $fallback;
    return;
}

// Build filename: character-uXXXX.svg (minimum 4 hex digits; longer for >FFFF)
$hex = strtolower(sprintf('%04x', $codepoint));
$filename = "character-u{$hex}.svg";

// Resolve and lock to sketches directory
$sketchesDir = rtrim(ROOT . DS . 'sketches', DS);
$path = $sketchesDir . DS . $filename;

// Ensure file exists and is inside sketchesDir (avoid traversal / weird symlinks)
$realSketchesDir = realpath($sketchesDir) ?: $sketchesDir;
$realPath = realpath($path);

if ($realPath === false || strncmp($realPath, $realSketchesDir . DS, strlen($realSketchesDir) + 1) !== 0 || !is_file($realPath)) {
    if ($strict) {
        throw new RuntimeException("SVG not found for '{$ch}' (expected: {$path}).");
    }
    echo $fallback;
    return;
}

// Output raw SVG markup (treat these SVGs as trusted local assets)
$svg = file_get_contents($realPath);
if ($svg === false) {
    if ($strict) {
        throw new RuntimeException("Failed to read SVG file: {$realPath}");
    }
    echo $fallback;
    return;
}

echo $svg;
