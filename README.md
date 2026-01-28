
# Ligatureluurs

Zoveel ligaturen dat je er tureluurs van wordt.

## Web Fonts

```css
@font-face {
  font-family: "Ligatureluurs";
  src:
    url("https://hetcdn.nl/fonts/ligatureluurs.woff2") format("woff2"),
    url("https://hetcdn.nl/fonts/ligatureluurs.woff") format("woff"),
    url("https://hetcdn.nl/fonts/ligatureluurs.ttf") format("truetype");
  font-weight: 400;
  font-style: normal;
  font-display: swap;
}

:root {
    --font-ligatureluurs: "Ligatureluurs", system-ui, sans-serif;
}

.font--ligatureluurs {
    font-family: var(--font-ligatureluurs);
}
```

## Letterproef

![Sheet.](sketches/sheet.svg)