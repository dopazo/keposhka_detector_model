# Keposhka Translator App

Especificación técnica para una aplicación móvil Android que traduce texto Keposhka en imágenes usando detección con YOLO, similar a Google Lens pero enfocada en el lenguaje ficticio Keposhka.

## Objetivo

Crear una aplicación móvil que permita:
1. Tomar o cargar fotos con texto Keposhka
2. Detectar y clasificar caracteres automáticamente (offline)
3. Mostrar la traducción superpuesta sobre cada carácter con blur de fondo
4. Comparar imagen original vs. traducida lado a lado
5. Mostrar referencia de caracteres para manejo de errores

## Stack Tecnológico

- **Framework**: Flutter 3.x
- **Lenguaje**: Dart
- **Inferencia ML**: TFLite (modelo YOLO exportado)
- **Cámara**: `camera` + `image_picker`
- **Procesamiento de imágenes**: `image` (dart)
- **Gestión de estado**: Riverpod 2.x
- **Navegación**: GoRouter
- **Almacenamiento local**: SharedPreferences

### Justificación de Flutter sobre React Native

| Criterio | Flutter | React Native |
|----------|---------|--------------|
| Rendimiento TFLite | Nativo via `tflite_flutter` | Requiere bridge JS, más latencia |
| Tamaño del bundle | ~15-20MB | ~25-35MB |
| Hot reload | Excelente | Bueno |
| Renderizado | Skia (consistente) | Componentes nativos (variación) |
| Complejidad TFLite | Baja (paquetes maduros) | Media (plugins nativos) |
| Procesamiento imagen | Dart `image` package | Necesita módulos nativos |

## Arquitectura de la App

### Estructura de Carpetas

```
keposhka_translator/
├── android/                    # Configuración Android
├── assets/
│   ├── fonts/
│   │   ├── Oswald-*.ttf       # Fuente principal UI
│   │   └── KeposhkaRegular.otf # Fuente Keposhka
│   ├── models/
│   │   └── keposhka_detector.tflite  # Modelo YOLO
│   └── images/
│       └── keposhka_reference/ # Imágenes de referencia (70 caracteres)
│           ├── 0.png
│           ├── 1.png
│           └── ... (A-Z, a-z, dígrafos)
│
├── lib/
│   ├── main.dart
│   ├── app.dart                # MaterialApp y tema
│   │
│   ├── core/
│   │   ├── theme/
│   │   │   ├── app_theme.dart      # Tema brutalista
│   │   │   ├── app_colors.dart     # Colores Keposhka
│   │   │   └── app_typography.dart # Tipografía
│   │   ├── constants/
│   │   │   └── class_labels.dart   # Mapeo de 70 clases
│   │   └── utils/
│   │       └── image_utils.dart    # Helpers de imagen
│   │
│   ├── features/
│   │   ├── camera/
│   │   │   ├── camera_screen.dart
│   │   │   └── camera_controller.dart
│   │   │
│   │   ├── translation/
│   │   │   ├── translation_screen.dart    # Pantalla principal de resultado
│   │   │   ├── translation_overlay.dart   # Widget de overlay con blur
│   │   │   └── translation_provider.dart  # Estado de traducción
│   │   │
│   │   ├── comparison/
│   │   │   ├── comparison_screen.dart     # Vista lado a lado
│   │   │   └── reference_grid.dart        # Grid de caracteres
│   │   │
│   │   └── home/
│   │       └── home_screen.dart           # Pantalla inicial
│   │
│   ├── models/
│   │   ├── detection.dart          # Clase Detection (box, class, conf)
│   │   └── translation_result.dart # Resultado completo
│   │
│   └── services/
│       ├── detector_service.dart   # Wrapper de TFLite
│       └── image_service.dart      # Procesamiento de imagen
│
├── test/
├── pubspec.yaml
└── README.md
```

### Flujo de Datos

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Cámara/   │────▶│  Preprocessing│────▶│  YOLO TFLite    │
│   Galería   │     │  (resize 640) │     │  Inference      │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                                                   ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  UI Result  │◀────│ Postprocessing│◀───│  Raw Detections │
│  (overlay)  │     │ (NMS, filter) │     │  (boxes+classes)│
└─────────────┘     └──────────────┘     └─────────────────┘
```

## Diseño Visual

### Sistema de Colores (Keposhka Regime)

```dart
// app_colors.dart
class AppColors {
  // Colores principales
  static const keposhkaRed = Color(0xFF89181A);
  static const keposhkaRedDark = Color(0xFF6B1315);
  static const keposhkaRedLight = Color(0xFFA51E21);

  // Neutrales
  static const keposhkaBlack = Color(0xFF000000);
  static const keposhkaWhite = Color(0xFFFFFFFF);
  static const keposhkaCream = Color(0xFFF5F0E6);
  static const keposhkaSteel = Color(0xFF2A2A2A);
  static const keposhkaConcrete = Color(0xFF1A1A1A);

  // Funcionales
  static const success = Color(0xFF2D5016);
  static const warning = Color(0xFF8B6914);
  static const error = keposhkaRed;
}
```

### Tipografía

```dart
// app_typography.dart
class AppTypography {
  // Títulos - Oswald Bold
  static const displayLarge = TextStyle(
    fontFamily: 'Oswald',
    fontSize: 32,
    fontWeight: FontWeight.w700,
    letterSpacing: 3.0,
  );

  // Subtítulos
  static const titleMedium = TextStyle(
    fontFamily: 'Oswald',
    fontSize: 18,
    fontWeight: FontWeight.w500,
    letterSpacing: 1.5,
  );

  // Texto de traducción superpuesto
  static const translationOverlay = TextStyle(
    fontFamily: 'Oswald',
    fontSize: 24,
    fontWeight: FontWeight.w700,
    color: AppColors.keposhkaCream,
    shadows: [
      Shadow(offset: Offset(2, 2), color: AppColors.keposhkaBlack),
    ],
  );

  // Texto Keposhka (para referencia)
  static const keposhkaText = TextStyle(
    fontFamily: 'Keposhka',
    fontSize: 32,
  );
}
```

### Componentes de UI Brutalistas

```dart
// Botón estilo brutal
class BrutalButton extends StatelessWidget {
  final String text;
  final VoidCallback onPressed;
  final bool isPrimary;

  // Estilo: bordes duros, sombra offset, sin bordes redondeados
  // Hover/press: sombra se reduce, botón se desplaza
}

// Panel con header de propaganda
class RegimePanel extends StatelessWidget {
  final String title;
  final Widget child;

  // Header rojo con patrón diagonal 45°
  // Borde inferior grueso negro
  // Tracking de letras amplio, uppercase
}

// Card con esquina cortada
class CutCornerCard extends StatelessWidget {
  // clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%)
}
```

## Funcionalidades Detalladas

### 1. Pantalla de Inicio (HomeScreen)

**Elementos:**
- Logo Keposhka (opcional)
- Botón "TOMAR FOTO" (abre cámara)
- Botón "SELECCIONAR IMAGEN" (abre galería)
- Enlace "VER REFERENCIA" (muestra grid de caracteres)

**Estilo:**
- Fondo: `keposhkaConcrete` (#1A1A1A)
- Botones: estilo brutal con sombra offset
- Texto: Oswald uppercase, tracking amplio

### 2. Pantalla de Cámara (CameraScreen)

**Elementos:**
- Preview de cámara fullscreen
- Botón de captura (círculo con borde rojo)
- Botón para cambiar a galería
- Guía visual (opcional, rectángulo punteado)

**Comportamiento:**
- Captura foto de alta resolución
- Navega a `TranslationScreen` con la imagen

### 3. Pantalla de Traducción (TranslationScreen)

**Elementos principales:**
- Imagen con overlay de traducción
- Botón "COMPARAR" (floating, esquina inferior)
- Indicador de caracteres detectados

**Overlay de traducción:**
```dart
// Para cada detección:
// 1. Aplicar blur gaussiano en el área del bounding box
// 2. Superponer la letra traducida centrada en el box
// 3. El tamaño de fuente se escala según el tamaño del box
```

**Algoritmo de overlay:**
```
Para cada detección (box, class_id, confidence):
  1. Extraer región del bounding box
  2. Aplicar GaussianBlur (sigma: 10-15)
  3. Dibujar rectángulo semi-transparente (opcional, para legibilidad)
  4. Calcular tamaño de fuente proporcional al alto del box
  5. Renderizar letra traducida (class_labels[class_id]) centrada
```

**Mapeo de clases a letras:**
```dart
// class_labels.dart
const classLabels = [
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  // 0-9
  'A', 'a', 'B', 'b', 'C', 'c', 'CH', 'ch',         // 10-17
  'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'GH', 'gh', // 18-27
  'H', 'h', 'I', 'i', 'J', 'j', 'K', 'k', 'L', 'l', // 28-37
  'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'Q', 'q', // 38-47
  'R', 'r', 'S', 's', 'SH', 'sh', 'T', 't', 'TH', 'th', // 48-57
  'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', // 58-67
  'Z', 'z',                                          // 68-69
];
```

### 4. Pantalla de Comparación (ComparisonScreen)

**Layout:**
```
┌────────────────────────────────────────┐
│         MODO COMPARACIÓN               │ ← Header rojo
├───────────────────┬────────────────────┤
│                   │                    │
│   ORIGINAL        │    TRADUCIDO       │
│   (imagen sin     │    (imagen con     │
│    procesar)      │     overlay)       │
│                   │                    │
├───────────────────┴────────────────────┤
│         REFERENCIA DE CARACTERES       │ ← Sección inferior
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ...   │
│  │ A │ │ B │ │ C │ │ D │ │ E │        │
│  │ 𐌀 │ │ 𐌁 │ │ 𐌂 │ │ 𐌃 │ │ 𐌄 │        │ ← Keposhka glyph
│  └───┘ └───┘ └───┘ └───┘ └───┘        │
└────────────────────────────────────────┘
```

**Elementos:**
- Dos imágenes lado a lado (scroll sincronizado, zoom sincronizado)
- Slider para ajustar división (opcional, modo overlay)
- Grid de referencia colapsable con las 70 clases
- Cada celda: letra latina arriba, glifo Keposhka abajo

**Interacciones:**
- Pinch to zoom (sincronizado en ambas imágenes)
- Pan (sincronizado)
- Tap en celda de referencia: resalta esa letra en la imagen traducida

### 5. Panel de Referencia (ReferenceGrid)

**Propósito:** Ayudar al usuario a verificar/corregir errores de detección

**Layout:**
- Grid de 7-8 columnas
- 70 celdas (una por clase)
- Cada celda muestra:
  - Letra latina (arriba, fuente Oswald)
  - Glifo Keposhka (abajo, fuente Keposhka o imagen PNG)

**Organización por grupos:**
```
DÍGITOS:     0-9 (10 celdas)
MAYÚSCULAS:  A-Z + CH, GH, SH, TH (30 celdas)
MINÚSCULAS:  a-z + ch, gh, sh, th (30 celdas)
```

## Servicio de Detección (TFLite)

### Configuración del Modelo

```dart
// detector_service.dart
class DetectorService {
  late Interpreter _interpreter;
  late List<String> _labels;

  // Configuración
  static const inputSize = 640;
  static const numClasses = 70;
  static const confidenceThreshold = 0.25;
  static const iouThreshold = 0.45;

  Future<void> initialize() async {
    // Cargar modelo desde assets
    _interpreter = await Interpreter.fromAsset('assets/models/keposhka_detector.tflite');
    _labels = classLabels;
  }

  Future<List<Detection>> detect(Uint8List imageBytes) async {
    // 1. Preprocesar imagen
    final input = _preprocessImage(imageBytes);

    // 2. Ejecutar inferencia
    final output = List.filled(1 * 8400 * (4 + numClasses), 0.0)
        .reshape([1, 8400, 4 + numClasses]);
    _interpreter.run(input, output);

    // 3. Postprocesar (NMS)
    return _postprocess(output);
  }

  Float32List _preprocessImage(Uint8List bytes) {
    // Decodificar imagen
    // Resize a 640x640 (letterbox)
    // Normalizar a [0, 1]
    // Convertir a Float32List [1, 640, 640, 3]
  }

  List<Detection> _postprocess(List output) {
    // Extraer boxes, scores, class_ids
    // Aplicar NMS
    // Filtrar por confidence threshold
    // Retornar lista de Detection
  }
}
```

### Modelo de Detección

```dart
// detection.dart
class Detection {
  final Rect boundingBox;  // Coordenadas normalizadas [0, 1]
  final int classId;       // 0-69
  final double confidence; // 0-1

  String get label => classLabels[classId];

  Rect getAbsoluteBox(Size imageSize) {
    return Rect.fromLTWH(
      boundingBox.left * imageSize.width,
      boundingBox.top * imageSize.height,
      boundingBox.width * imageSize.width,
      boundingBox.height * imageSize.height,
    );
  }
}
```

## Procesamiento de Imagen

### Aplicar Overlay de Traducción

```dart
// image_service.dart
class ImageService {
  /// Genera imagen con traducción superpuesta
  Future<Uint8List> applyTranslationOverlay(
    Uint8List originalImage,
    List<Detection> detections,
  ) async {
    final image = img.decodeImage(originalImage)!;

    for (final detection in detections) {
      final box = detection.getAbsoluteBox(
        Size(image.width.toDouble(), image.height.toDouble())
      );

      // 1. Extraer región
      final region = img.copyCrop(
        image,
        x: box.left.toInt(),
        y: box.top.toInt(),
        width: box.width.toInt(),
        height: box.height.toInt(),
      );

      // 2. Aplicar blur
      final blurred = img.gaussianBlur(region, radius: 12);

      // 3. Pegar región con blur
      img.compositeImage(
        image, blurred,
        dstX: box.left.toInt(),
        dstY: box.top.toInt(),
      );

      // 4. Dibujar letra traducida
      _drawText(
        image,
        detection.label,
        box.center,
        fontSize: (box.height * 0.8).toInt(),
      );
    }

    return Uint8List.fromList(img.encodePng(image));
  }
}
```

## Configuración de Dependencias

### pubspec.yaml

```yaml
name: keposhka_translator
description: Translator app for Keposhka fictional language
version: 1.0.0+1

environment:
  sdk: '>=3.0.0 <4.0.0'

dependencies:
  flutter:
    sdk: flutter

  # Inferencia ML
  tflite_flutter: ^0.10.4

  # Cámara y galería
  camera: ^0.11.0
  image_picker: ^1.0.7

  # Procesamiento de imagen
  image: ^4.1.7

  # Estado
  flutter_riverpod: ^2.5.1

  # Navegación
  go_router: ^13.2.0

  # UI
  flutter_svg: ^2.0.10+1  # Para iconos SVG

  # Almacenamiento
  shared_preferences: ^2.2.2

  # Utilidades
  path_provider: ^2.1.2
  permission_handler: ^11.3.0

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^3.0.0

flutter:
  uses-material-design: true

  fonts:
    - family: Oswald
      fonts:
        - asset: assets/fonts/Oswald-Regular.ttf
        - asset: assets/fonts/Oswald-Bold.ttf
          weight: 700
        - asset: assets/fonts/Oswald-Medium.ttf
          weight: 500
    - family: Keposhka
      fonts:
        - asset: assets/fonts/KeposhkaRegular.otf

  assets:
    - assets/models/keposhka_detector.tflite
    - assets/images/keposhka_reference/
```

## Permisos Android

### android/app/src/main/AndroidManifest.xml

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <!-- Cámara -->
    <uses-permission android:name="android.permission.CAMERA" />
    <uses-feature android:name="android.hardware.camera" android:required="true" />
    <uses-feature android:name="android.hardware.camera.autofocus" android:required="false" />

    <!-- Almacenamiento (para galería) -->
    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"
                     android:maxSdkVersion="32" />
    <uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />

    <application
        android:label="Keposhka Translator"
        android:name="${applicationName}"
        android:icon="@mipmap/ic_launcher">
        <!-- ... -->
    </application>
</manifest>
```

## Consideraciones de Rendimiento

### Inferencia TFLite

| Aspecto | Recomendación |
|---------|---------------|
| Delegado | GPU delegate para mejor rendimiento |
| Threads | 4 threads para CPU fallback |
| Modelo | FP16 (~8MB) balanceo tamaño/precisión |
| Batch size | 1 (inferencia por imagen) |
| Input | 640x640 RGB normalizado |

### Optimizaciones

1. **Carga lazy del modelo**: Inicializar TFLite solo al entrar a cámara/galería
2. **Procesamiento en isolate**: Ejecutar inferencia en isolate separado
3. **Cache de resultados**: Guardar detecciones mientras no cambie la imagen
4. **Imagen reducida para preview**: Usar thumbnail para UI, full-res para export

## Fases de Desarrollo

### Fase 1: Setup y UI Base
- [ ] Crear proyecto Flutter
- [ ] Configurar tema brutalista (colores, tipografía)
- [ ] Implementar componentes base (BrutalButton, RegimePanel, etc.)
- [ ] Crear navegación con GoRouter
- [ ] Implementar HomeScreen
- [ ] Agregar fuentes (Oswald, Keposhka)

### Fase 2: Captura de Imagen
- [ ] Integrar plugin de cámara
- [ ] Implementar CameraScreen
- [ ] Integrar image_picker para galería
- [ ] Manejar permisos de cámara y almacenamiento
- [ ] Navegar a TranslationScreen con imagen

### Fase 3: Integración TFLite
- [ ] Copiar modelo TFLite a assets
- [ ] Implementar DetectorService
- [ ] Configurar preprocesamiento de imagen
- [ ] Implementar postprocesamiento (NMS)
- [ ] Probar inferencia básica

### Fase 4: Overlay de Traducción
- [ ] Implementar ImageService para blur
- [ ] Renderizar letras sobre blur
- [ ] Crear TranslationScreen con imagen procesada
- [ ] Ajustar tamaño de fuente según box
- [ ] Optimizar rendimiento (isolate)

### Fase 5: Modo Comparación
- [ ] Implementar ComparisonScreen
- [ ] Vista lado a lado sincronizada
- [ ] Implementar ReferenceGrid
- [ ] Preparar imágenes de referencia Keposhka
- [ ] Scroll/zoom sincronizado

### Fase 6: Pulido y Testing
- [ ] Pruebas en múltiples dispositivos
- [ ] Optimizar tiempos de carga
- [ ] Manejar edge cases (sin detecciones, errores)
- [ ] Agregar animaciones y transiciones
- [ ] Testing de UI y UX

### Fase 7: Release
- [ ] Configurar firma de APK
- [ ] Generar APK release
- [ ] Probar en dispositivo físico
- [ ] Documentar uso e instalación
- [ ] (Opcional) Preparar para Play Store

## Métricas de UX Objetivo

| Métrica | Objetivo |
|---------|----------|
| Tiempo de carga inicial | < 2s |
| Tiempo de inferencia | < 500ms (GPU), < 1.5s (CPU) |
| Tiempo total foto→traducción | < 3s |
| Tamaño APK | < 50MB |
| RAM en uso | < 200MB |

## Troubleshooting

### TFLite no carga en release
```
Asegurar que el modelo esté en:
android/app/src/main/assets/models/keposhka_detector.tflite

Y en pubspec.yaml:
  assets:
    - assets/models/
```

### Cámara no inicia
```
Verificar permisos en AndroidManifest.xml
Usar permission_handler para solicitar permisos en runtime
```

### Inferencia lenta
```
1. Verificar que GPU delegate está activo
2. Reducir tamaño de entrada si es aceptable
3. Mover inferencia a isolate
4. Considerar modelo más pequeño (yolo11n)
```

---

## Referencias

- Modelo YOLO: `keposhka-detector-model/specs/keposhka_detector_model.md`
- Estilo UI: `keposhka_labeling/frontend/src/index.css`
- TFLite Flutter: https://pub.dev/packages/tflite_flutter
- YOLO TFLite export: Ultralytics docs
