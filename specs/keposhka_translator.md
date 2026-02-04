# Keposhka Translator App

Especificación técnica para una aplicación móvil Android que traduce texto Keposhka en imágenes usando detección con YOLO, similar a Google Lens pero enfocada en el lenguaje ficticio Keposhka.

## Objetivo

Crear una aplicación móvil que permita:
1. Tomar o cargar fotos con texto Keposhka
2. Detectar y clasificar caracteres automáticamente (offline, 100% sin conexión)
3. Mostrar la traducción superpuesta sobre cada carácter con blur de fondo
4. Agrupar caracteres cercanos en palabras para mejor legibilidad
5. Verificar traducción comparando imagen original vs. traducida lado a lado
6. Mostrar referencia de caracteres mediante carrusel inferior
7. Permitir corrección manual de detecciones incorrectas

## Stack Tecnológico

- **Framework**: Flutter 3.x
- **Lenguaje**: Dart
- **Inferencia ML**: TFLite (modelo YOLO exportado)
- **Cámara**: `camera` + `image_picker`
- **Procesamiento de imágenes**: `image` (dart)
- **Gestión de estado**: Riverpod 2.x
- **Navegación**: GoRouter
- **Almacenamiento local**: SharedPreferences

## Arquitectura de la App

### Estructura de Carpetas

```
keposhka_translator/
├── android/                    # Configuración Android
├── assets/
│   ├── fonts/
│   │   ├── Oswald-*.ttf       # Fuente principal UI
│   │   └── KeposhkaRegular.otf # Fuente Keposhka (incluye todos los glifos)
│   └── models/
│       └── keposhka_detector.tflite  # Modelo YOLO
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
│   │   ├── verification/
│   │   │   ├── verification_screen.dart     # Vista lado a lado
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

### Tema: Brutalismo/Distópico con Influencia Soviética

**Concepto:** Interfaz institucional, rígida y opresiva, como una agencia de policía secreta catalogando un idioma ficticio. Inspirado en propaganda constructivista soviética y estética industrial.

| Aspecto | Descripción |
|---------|-------------|
| **Colores** | Rojo propagandístico (#89181A) + negro + crema |
| **Tipografía** | Pesada e industrial con espaciado amplio (tracking) |
| **Sombras** | Offset brutal (4px) en vez de drops suaves |
| **Bordes** | Gruesos, sólidos, sin border-radius |
| **Texturas** | Líneas de escaneo, rayas diagonales 45°, ruido |
| **Animaciones** | Escaneo, parpadeo CRT, stamp "APROBADO" |
| **Interacción** | Botones con efecto "punch" (se hunden al presionar) |

### Sistema de Colores (Draag Regime Palette)

```dart
// app_colors.dart
class AppColors {
  // === Colores principales ===
  static const keposhkaRed = Color(0xFF89181A);      // Rojo propagandístico
  static const keposhkaRedDark = Color(0xFF6B1315); // Hover/pressed
  static const keposhkaRedLight = Color(0xFFA51E21); // Accent

  // === Neutrales ===
  static const keposhkaBlack = Color(0xFF000000);
  static const keposhkaWhite = Color(0xFFFFFFFF);
  static const keposhkaCream = Color(0xFFF5F0E6);    // Texto principal sobre fondo oscuro
  static const keposhkaSteel = Color(0xFF2A2A2A);    // Bordes, elementos secundarios
  static const keposhkaConcrete = Color(0xFF1A1A1A); // Fondo principal
  static const keposhkaRust = Color(0xFF4A1A1B);     // Accent oscuro

  // === Funcionales ===
  static const success = Color(0xFF2D5016);          // Verde oscuro militar
  static const successLight = Color(0xFF4A7C23);
  static const warning = Color(0xFF8B6914);
  static const error = keposhkaRed;
}
```

### Tipografía

**Fuentes requeridas:**
- **Bebas Neue** - Display/títulos grandes (constructivista)
- **Oswald** - Headings y UI general (industrial)
- **JetBrains Mono** - Código, datos, IDs (monospace)
- **Keposhka** - Glifos del idioma ficticio

```dart
// app_typography.dart
class AppTypography {
  // === Display - Bebas Neue (títulos grandes, propaganda) ===
  static const displayLarge = TextStyle(
    fontFamily: 'BebasNeue',
    fontSize: 48,
    fontWeight: FontWeight.w400,
    letterSpacing: 4.0,  // Tracking amplio
    height: 1.0,
  );

  // === Headings - Oswald (subtítulos, labels) ===
  static const headingLarge = TextStyle(
    fontFamily: 'Oswald',
    fontSize: 24,
    fontWeight: FontWeight.w700,
    letterSpacing: 3.0,
  );

  static const headingMedium = TextStyle(
    fontFamily: 'Oswald',
    fontSize: 18,
    fontWeight: FontWeight.w500,
    letterSpacing: 2.0,
  );

  static const headingSmall = TextStyle(
    fontFamily: 'Oswald',
    fontSize: 14,
    fontWeight: FontWeight.w500,
    letterSpacing: 1.5,
  );

  // === Monospace - JetBrains Mono (datos, IDs, código) ===
  static const mono = TextStyle(
    fontFamily: 'JetBrainsMono',
    fontSize: 12,
    fontWeight: FontWeight.w400,
    letterSpacing: 0,
  );

  static const monoSmall = TextStyle(
    fontFamily: 'JetBrainsMono',
    fontSize: 10,
    fontWeight: FontWeight.w400,
  );

  // === Texto de traducción superpuesto ===
  static const translationOverlay = TextStyle(
    fontFamily: 'Oswald',
    fontSize: 24,
    fontWeight: FontWeight.w700,
    color: AppColors.keposhkaCream,
    shadows: [
      Shadow(offset: Offset(2, 2), blurRadius: 0, color: AppColors.keposhkaBlack),
    ],
  );

  // === Keposhka (glifos del idioma) ===
  static const keposhkaLarge = TextStyle(
    fontFamily: 'Keposhka',
    fontSize: 32,
  );

  static const keposhkaMedium = TextStyle(
    fontFamily: 'Keposhka',
    fontSize: 24,
  );
}
```

### Sombras Brutales

```dart
// app_shadows.dart
class AppShadows {
  // Sombra brutal estándar (4px offset, sin blur)
  static const brutal = BoxShadow(
    color: AppColors.keposhkaRed,
    offset: Offset(4, 4),
    blurRadius: 0,
    spreadRadius: 0,
  );

  // Sombra brutal pequeña
  static const brutalSm = BoxShadow(
    color: AppColors.keposhkaRed,
    offset: Offset(2, 2),
    blurRadius: 0,
    spreadRadius: 0,
  );

  // Sombra brutal negra (para botones secundarios)
  static const brutalBlack = BoxShadow(
    color: AppColors.keposhkaBlack,
    offset: Offset(4, 4),
    blurRadius: 0,
    spreadRadius: 0,
  );

  // Inner glow (para panels activos)
  static const innerGlow = BoxShadow(
    color: Color(0x4D89181A), // keposhkaRed con 30% opacity
    offset: Offset.zero,
    blurRadius: 20,
    spreadRadius: -5,
  );
}
```

### Animaciones

```dart
// lib/core/theme/app_animations.dart

import 'dart:math';
import 'package:flutter/material.dart';

/// Efecto de escaneo vertical (como monitor CRT)
/// Línea semi-transparente que baja por la pantalla
class ScanlineEffect extends StatefulWidget {
  final Widget child;
  final Duration duration;

  const ScanlineEffect({
    super.key,
    required this.child,
    this.duration = const Duration(seconds: 8),
  });

  @override
  State<ScanlineEffect> createState() => _ScanlineEffectState();
}

class _ScanlineEffectState extends State<ScanlineEffect>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: widget.duration)
      ..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        widget.child,
        Positioned.fill(
          child: IgnorePointer(
            child: AnimatedBuilder(
              animation: _controller,
              builder: (context, _) {
                return CustomPaint(
                  painter: _ScanlinePainter(progress: _controller.value),
                );
              },
            ),
          ),
        ),
      ],
    );
  }
}

class _ScanlinePainter extends CustomPainter {
  final double progress;

  _ScanlinePainter({required this.progress});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withOpacity(0.03)
      ..strokeWidth = 2;

    final y = size.height * progress;
    canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
  }

  @override
  bool shouldRepaint(_ScanlinePainter oldDelegate) =>
      oldDelegate.progress != progress;
}

/// Efecto de parpadeo sutil (CRT flicker)
/// Opacidad oscila entre 1.0 y 0.95
class FlickerEffect extends StatefulWidget {
  final Widget child;

  const FlickerEffect({super.key, required this.child});

  @override
  State<FlickerEffect> createState() => _FlickerEffectState();
}

class _FlickerEffectState extends State<FlickerEffect>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _opacity;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 150),
    )..repeat(reverse: true);

    _opacity = Tween<double>(begin: 1.0, end: 0.95).animate(_controller);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(opacity: _opacity, child: widget.child);
  }
}

/// Efecto de sello "APROBADO" / "TRADUCIDO"
/// Escala: 1.5 → 0.95 → 1.0, Rotación: -15° → -12°
class StampEffect extends StatefulWidget {
  final String text;
  final bool show;
  final VoidCallback? onComplete;

  const StampEffect({
    super.key,
    this.text = 'APROBADO',
    this.show = false,
    this.onComplete,
  });

  @override
  State<StampEffect> createState() => _StampEffectState();
}

class _StampEffectState extends State<StampEffect>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scale;
  late Animation<double> _rotation;
  late Animation<double> _opacity;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 300),
    );

    // Escala: 1.5 → 0.95 → 1.0
    _scale = TweenSequence<double>([
      TweenSequenceItem(tween: Tween(begin: 1.5, end: 0.95), weight: 70),
      TweenSequenceItem(tween: Tween(begin: 0.95, end: 1.0), weight: 30),
    ]).animate(CurvedAnimation(parent: _controller, curve: Curves.easeOut));

    // Rotación: -15° → -12°
    _rotation = Tween<double>(
      begin: -15 * pi / 180,
      end: -12 * pi / 180,
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeOut));

    // Opacidad: 0 → 1
    _opacity = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(parent: _controller, curve: const Interval(0, 0.7)),
    );

    _controller.addStatusListener((status) {
      if (status == AnimationStatus.completed) {
        widget.onComplete?.call();
      }
    });
  }

  @override
  void didUpdateWidget(StampEffect oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.show && !oldWidget.show) {
      _controller.forward(from: 0);
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        if (_opacity.value == 0) return const SizedBox.shrink();

        return Opacity(
          opacity: _opacity.value,
          child: Transform.scale(
            scale: _scale.value,
            child: Transform.rotate(
              angle: _rotation.value,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  border: Border.all(color: AppColors.keposhkaRed, width: 3),
                  color: Colors.transparent,
                ),
                child: Text(
                  widget.text,
                  style: AppTypography.headingLarge.copyWith(
                    color: AppColors.keposhkaRed,
                  ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}
```

### Texturas y Efectos

```dart
// lib/core/theme/app_decorations.dart

import 'dart:math';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';

/// Patrón de rayas diagonales soviéticas (45°)
/// Equivalente a: repeating-linear-gradient(45deg, transparent 0 2px, red/10% 2px 4px)
class DiagonalStripesOverlay extends StatelessWidget {
  final Widget child;
  final Color color;
  final double stripeWidth;
  final double gapWidth;

  const DiagonalStripesOverlay({
    super.key,
    required this.child,
    this.color = const Color(0x1A89181A), // keposhkaRed con 10% opacity
    this.stripeWidth = 2,
    this.gapWidth = 2,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        child,
        Positioned.fill(
          child: IgnorePointer(
            child: CustomPaint(
              painter: _DiagonalStripesPainter(
                color: color,
                stripeWidth: stripeWidth,
                gapWidth: gapWidth,
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class _DiagonalStripesPainter extends CustomPainter {
  final Color color;
  final double stripeWidth;
  final double gapWidth;

  _DiagonalStripesPainter({
    required this.color,
    required this.stripeWidth,
    required this.gapWidth,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = stripeWidth
      ..style = PaintingStyle.stroke;

    final step = stripeWidth + gapWidth;
    final diagonal = sqrt(size.width * size.width + size.height * size.height);

    // Dibujar líneas diagonales 45°
    for (double i = -diagonal; i < diagonal * 2; i += step) {
      canvas.drawLine(
        Offset(i, 0),
        Offset(i + size.height, size.height),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(_DiagonalStripesPainter oldDelegate) =>
      oldDelegate.color != color ||
      oldDelegate.stripeWidth != stripeWidth ||
      oldDelegate.gapWidth != gapWidth;
}

/// Overlay de ruido (noise texture)
/// Simula textura granulada industrial
class NoiseOverlay extends StatelessWidget {
  final Widget child;
  final double opacity;

  const NoiseOverlay({
    super.key,
    required this.child,
    this.opacity = 0.03,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        child,
        Positioned.fill(
          child: IgnorePointer(
            child: Opacity(
              opacity: opacity,
              child: CustomPaint(
                painter: _NoisePainter(),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class _NoisePainter extends CustomPainter {
  final Random _random = Random(42); // Seed fijo para consistencia

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint();
    const pixelSize = 2.0;

    for (double x = 0; x < size.width; x += pixelSize) {
      for (double y = 0; y < size.height; y += pixelSize) {
        final gray = _random.nextInt(256);
        paint.color = Color.fromARGB(255, gray, gray, gray);
        canvas.drawRect(
          Rect.fromLTWH(x, y, pixelSize, pixelSize),
          paint,
        );
      }
    }
  }

  @override
  bool shouldRepaint(_NoisePainter oldDelegate) => false;
}

/// Líneas de escaneo horizontales (CRT effect)
/// Equivalente a: repeating-linear-gradient(0deg, transparent 0 2px, black/10% 2px 4px)
class ScanlinesOverlay extends StatelessWidget {
  final Widget child;
  final double lineSpacing;
  final double opacity;

  const ScanlinesOverlay({
    super.key,
    required this.child,
    this.lineSpacing = 4,
    this.opacity = 0.1,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        child,
        Positioned.fill(
          child: IgnorePointer(
            child: CustomPaint(
              painter: _ScanlinesPainter(
                lineSpacing: lineSpacing,
                opacity: opacity,
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class _ScanlinesPainter extends CustomPainter {
  final double lineSpacing;
  final double opacity;

  _ScanlinesPainter({required this.lineSpacing, required this.opacity});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.black.withOpacity(opacity)
      ..strokeWidth = 1;

    for (double y = 0; y < size.height; y += lineSpacing) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  @override
  bool shouldRepaint(_ScanlinesPainter oldDelegate) =>
      oldDelegate.lineSpacing != lineSpacing || oldDelegate.opacity != opacity;
}

/// Esquina cortada (cut corner effect)
/// Equivalente a: clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%)
class CutCornerClipper extends CustomClipper<Path> {
  final double cutSize;

  CutCornerClipper({this.cutSize = 12});

  @override
  Path getClip(Size size) {
    return Path()
      ..moveTo(0, 0)
      ..lineTo(size.width - cutSize, 0)
      ..lineTo(size.width, cutSize)
      ..lineTo(size.width, size.height)
      ..lineTo(0, size.height)
      ..close();
  }

  @override
  bool shouldReclip(CutCornerClipper oldClipper) =>
      oldClipper.cutSize != cutSize;
}

/// Widget helper para aplicar esquina cortada
class CutCornerContainer extends StatelessWidget {
  final Widget child;
  final double cutSize;
  final Color? backgroundColor;
  final Color? borderColor;
  final double borderWidth;

  const CutCornerContainer({
    super.key,
    required this.child,
    this.cutSize = 12,
    this.backgroundColor,
    this.borderColor,
    this.borderWidth = 2,
  });

  @override
  Widget build(BuildContext context) {
    return ClipPath(
      clipper: CutCornerClipper(cutSize: cutSize),
      child: Container(
        decoration: BoxDecoration(
          color: backgroundColor ?? AppColors.keposhkaConcrete,
        ),
        child: CustomPaint(
          painter: _CutCornerBorderPainter(
            cutSize: cutSize,
            borderColor: borderColor ?? AppColors.keposhkaRed,
            borderWidth: borderWidth,
          ),
          child: child,
        ),
      ),
    );
  }
}

class _CutCornerBorderPainter extends CustomPainter {
  final double cutSize;
  final Color borderColor;
  final double borderWidth;

  _CutCornerBorderPainter({
    required this.cutSize,
    required this.borderColor,
    required this.borderWidth,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = borderColor
      ..strokeWidth = borderWidth
      ..style = PaintingStyle.stroke;

    final path = Path()
      ..moveTo(0, 0)
      ..lineTo(size.width - cutSize, 0)
      ..lineTo(size.width, cutSize)
      ..lineTo(size.width, size.height)
      ..lineTo(0, size.height)
      ..close();

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(_CutCornerBorderPainter oldDelegate) =>
      oldDelegate.cutSize != cutSize ||
      oldDelegate.borderColor != borderColor ||
      oldDelegate.borderWidth != borderWidth;
}

/// Wrapper que combina múltiples efectos (noise + scanlines)
/// Usar para contenedores principales como el body de RegimePanel
class RegimeEffectsOverlay extends StatelessWidget {
  final Widget child;
  final bool enableNoise;
  final bool enableScanlines;

  const RegimeEffectsOverlay({
    super.key,
    required this.child,
    this.enableNoise = true,
    this.enableScanlines = false,
  });

  @override
  Widget build(BuildContext context) {
    Widget result = child;

    if (enableNoise) {
      result = NoiseOverlay(opacity: 0.03, child: result);
    }

    if (enableScanlines) {
      result = ScanlinesOverlay(child: result);
    }

    return result;
  }
}
```

### Componentes de UI Brutalistas

```dart
// lib/core/widgets/brutal_button.dart

import 'package:flutter/material.dart';

/// Botón estilo brutal con efecto "punch"
/// Se hunde al presionar (sombra desaparece, botón se desplaza)
class BrutalButton extends StatefulWidget {
  final String text;
  final VoidCallback? onPressed;
  final bool isPrimary;
  final IconData? icon;
  final double shadowOffset;

  const BrutalButton({
    super.key,
    required this.text,
    this.onPressed,
    this.isPrimary = true,
    this.icon,
    this.shadowOffset = 4,
  });

  @override
  State<BrutalButton> createState() => _BrutalButtonState();
}

class _BrutalButtonState extends State<BrutalButton> {
  bool _isPressed = false;

  @override
  Widget build(BuildContext context) {
    final backgroundColor = widget.isPrimary
        ? AppColors.keposhkaRed
        : AppColors.keposhkaBlack;
    final textColor = AppColors.keposhkaCream;

    return GestureDetector(
      onTapDown: (_) => setState(() => _isPressed = true),
      onTapUp: (_) {
        setState(() => _isPressed = false);
        widget.onPressed?.call();
      },
      onTapCancel: () => setState(() => _isPressed = false),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 100),
        transform: Matrix4.translationValues(
          _isPressed ? widget.shadowOffset : 0,
          _isPressed ? widget.shadowOffset : 0,
          0,
        ),
        decoration: BoxDecoration(
          color: backgroundColor,
          border: Border.all(color: AppColors.keposhkaBlack, width: 2),
          boxShadow: _isPressed
              ? []
              : [
                  BoxShadow(
                    color: AppColors.keposhkaBlack,
                    offset: Offset(widget.shadowOffset, widget.shadowOffset),
                    blurRadius: 0,
                  ),
                ],
        ),
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (widget.icon != null) ...[
              Icon(widget.icon, color: textColor, size: 20),
              const SizedBox(width: 8),
            ],
            Text(
              widget.text.toUpperCase(),
              style: AppTypography.headingMedium.copyWith(
                color: textColor,
                letterSpacing: 2,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// lib/core/widgets/regime_panel.dart

/// Panel con header de propaganda estilo soviético
class RegimePanel extends StatelessWidget {
  final String title;
  final Widget child;
  final String? badge;
  final IconData? icon;
  final String? subtitle;

  const RegimePanel({
    super.key,
    required this.title,
    required this.child,
    this.badge,
    this.icon,
    this.subtitle,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // === HEADER ===
        Container(
          decoration: const BoxDecoration(
            color: AppColors.keposhkaRed,
            border: Border(
              bottom: BorderSide(color: AppColors.keposhkaBlack, width: 4),
            ),
          ),
          child: DiagonalStripesOverlay(
            color: AppColors.keposhkaBlack.withOpacity(0.1),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              child: Row(
                children: [
                  // Icono cuadrado
                  if (icon != null)
                    Container(
                      width: 32,
                      height: 32,
                      margin: const EdgeInsets.only(right: 12),
                      decoration: BoxDecoration(
                        color: AppColors.keposhkaBlack,
                        border: Border.all(
                          color: AppColors.keposhkaCream.withOpacity(0.2),
                        ),
                      ),
                      child: Icon(icon, color: AppColors.keposhkaCream, size: 18),
                    ),
                  // Título y subtítulo
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          title.toUpperCase(),
                          style: AppTypography.headingMedium.copyWith(
                            color: AppColors.keposhkaWhite,
                            letterSpacing: 3,
                          ),
                        ),
                        if (subtitle != null)
                          Text(
                            subtitle!.toUpperCase(),
                            style: AppTypography.monoSmall.copyWith(
                              color: AppColors.keposhkaCream.withOpacity(0.6),
                              letterSpacing: 2,
                            ),
                          ),
                      ],
                    ),
                  ),
                  // Badge
                  if (badge != null)
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: AppColors.keposhkaBlack,
                        border: Border.all(
                          color: AppColors.keposhkaCream.withOpacity(0.3),
                        ),
                      ),
                      child: Text(
                        badge!,
                        style: AppTypography.monoSmall.copyWith(
                          color: AppColors.keposhkaCream,
                          letterSpacing: 1,
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ),
        // === BODY ===
        Expanded(
          child: Container(
            decoration: const BoxDecoration(
              color: AppColors.keposhkaConcrete,
              border: Border(
                left: BorderSide(color: AppColors.keposhkaBlack, width: 4),
              ),
            ),
            child: NoiseOverlay(
              opacity: 0.03,
              child: child,
            ),
          ),
        ),
      ],
    );
  }
}

// lib/core/widgets/regime_badge.dart

/// Badge de clasificación estilo documento oficial
class RegimeBadge extends StatelessWidget {
  final String text;
  final Color? backgroundColor;
  final Color? textColor;

  const RegimeBadge({
    super.key,
    required this.text,
    this.backgroundColor,
    this.textColor,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: backgroundColor ?? AppColors.keposhkaRed,
        border: Border.all(color: AppColors.keposhkaBlack),
      ),
      child: Text(
        text.toUpperCase(),
        style: AppTypography.monoSmall.copyWith(
          color: textColor ?? AppColors.keposhkaWhite,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }
}

// lib/core/widgets/file_list_item.dart

/// Item de lista estilo archivo clasificado
class FileListItem extends StatelessWidget {
  final String filename;
  final bool isActive;
  final bool isDone;
  final VoidCallback? onTap;

  const FileListItem({
    super.key,
    required this.filename,
    this.isActive = false,
    this.isDone = false,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 100),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: isActive
              ? AppColors.keposhkaRed
              : AppColors.keposhkaSteel.withOpacity(0.3),
          border: Border(
            left: BorderSide(
              color: isActive
                  ? AppColors.keposhkaWhite
                  : AppColors.keposhkaRed,
              width: 4,
            ),
          ),
        ),
        child: Row(
          children: [
            Expanded(
              child: Text(
                filename,
                style: AppTypography.mono.copyWith(
                  color: isActive
                      ? AppColors.keposhkaWhite
                      : AppColors.keposhkaCream,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            if (isDone)
              Text(
                '✓',
                style: TextStyle(
                  color: isActive
                      ? AppColors.keposhkaWhite
                      : AppColors.keposhkaRed,
                  fontWeight: FontWeight.bold,
                ),
              ),
          ],
        ),
      ),
    );
  }
}

// lib/core/widgets/cut_corner_card.dart

/// Card con esquina cortada estilo industrial
class CutCornerCard extends StatelessWidget {
  final Widget child;
  final double cutSize;
  final Color? backgroundColor;
  final bool showShadow;

  const CutCornerCard({
    super.key,
    required this.child,
    this.cutSize = 12,
    this.backgroundColor,
    this.showShadow = true,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: showShadow
          ? const BoxDecoration(
              boxShadow: [
                BoxShadow(
                  color: AppColors.keposhkaBlack,
                  offset: Offset(4, 4),
                  blurRadius: 0,
                ),
              ],
            )
          : null,
      child: CutCornerContainer(
        cutSize: cutSize,
        backgroundColor: backgroundColor ?? AppColors.keposhkaConcrete,
        borderColor: AppColors.keposhkaRed,
        borderWidth: 2,
        child: child,
      ),
    );
  }
}

// lib/core/widgets/industrial_input.dart

/// Input field estilo industrial/terminal
class IndustrialInput extends StatelessWidget {
  final TextEditingController? controller;
  final String? hintText;
  final ValueChanged<String>? onChanged;
  final bool readOnly;

  const IndustrialInput({
    super.key,
    this.controller,
    this.hintText,
    this.onChanged,
    this.readOnly = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.keposhkaBlack,
        border: Border.all(color: AppColors.keposhkaSteel, width: 2),
      ),
      child: TextField(
        controller: controller,
        onChanged: onChanged,
        readOnly: readOnly,
        style: AppTypography.mono.copyWith(color: AppColors.keposhkaCream),
        decoration: InputDecoration(
          hintText: hintText,
          hintStyle: AppTypography.mono.copyWith(color: AppColors.keposhkaSteel),
          contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          border: InputBorder.none,
          focusedBorder: InputBorder.none,
        ),
        cursorColor: AppColors.keposhkaRed,
      ),
    );
  }
}
```

### Archivos de Referencia de Diseño

Los estilos están basados en el proyecto `keposhka_labeling`:

```
keposhka_labeling/frontend/
├── tailwind.config.js    # Colores, fuentes, sombras, animaciones
├── src/index.css         # Clases CSS, componentes, keyframes
└── package.json          # Fuentes de Google (Bebas Neue, Oswald)
```

**Fuentes en pubspec.yaml:**
```yaml
flutter:
  fonts:
    - family: BebasNeue
      fonts:
        - asset: assets/fonts/BebasNeue-Regular.ttf
    - family: Oswald
      fonts:
        - asset: assets/fonts/Oswald-Regular.ttf
        - asset: assets/fonts/Oswald-Medium.ttf
          weight: 500
        - asset: assets/fonts/Oswald-Bold.ttf
          weight: 700
    - family: JetBrainsMono
      fonts:
        - asset: assets/fonts/JetBrainsMono-Regular.ttf
    - family: Keposhka
      fonts:
        - asset: assets/fonts/KeposhkaRegular.otf
```

## Funcionalidades Detalladas

### 1. Pantalla de Inicio (HomeScreen)

**Elementos:**
- Logo Keposhka (opcional) ("Keposhka Translator" con fuente y letras en Keposhka)
- Botón "TOMAR FOTO" (abre cámara)
- Botón "SELECCIONAR IMAGEN" (abre galería)
- Sección "RECIENTES" (lista horizontal de thumbnails de últimas 10-20 traducciones)
- Toggle para mostrar/ocultar carrusel de referencia (franja inferior)

**Historial temporal:**
- Guarda las últimas 10-20 traducciones en caché local
- Se muestra como lista horizontal de thumbnails debajo de los botones principales
- Tap en thumbnail abre la traducción guardada
- Se borran eventualmente (no persistencia permanente)

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
- Botón "VERIFICAR" (floating, esquina inferior)
- Indicador de caracteres detectados (solo conteo total: "42 caracteres detectados")
- Botón de guardar foto (guardar original, guardar traducción a galería del sistema en formato PNG)
- Carrusel de referencia inferior (toggle para mostrar/ocultar, ~80px de altura)

**Color coding de confianza:**
Las letras traducidas usan colores según la confianza del modelo:
- Verde: confianza > 80%
- Amarillo: confianza 50-80%
- Rojo: confianza < 50% (pero >= threshold de 25%)

**Overlay de traducción:**
```dart
// Para cada detección:
// 1. Aplicar blur gaussiano en el área del bounding box
// 2. Superponer la letra traducida (grande) + glifo Keposhka original (pequeño debajo)
// 3. El tamaño de fuente se escala según el tamaño del box
// 4. Color según nivel de confianza (verde/amarillo/rojo)
```

**Formato de overlay por carácter:**
- Letra latina traducida: fuente grande, centrada en el box, color según confianza
- Glifo Keposhka detectado: fuente pequeña debajo de la latina, ayuda a confirmar la detección

**Algoritmo de agrupación de palabras:**
```
1. Ordenar detecciones por posición X
2. Para cada par de detecciones adyacentes:
   - Calcular distancia horizontal entre bounding boxes
   - Si distancia < 50% del ancho promedio de ambos caracteres:
     - Agrupar en la misma palabra
3. Para cada grupo/palabra:
   - Calcular bounding box unificado que englobe todos los caracteres
   - Aplicar blur gaussiano a toda la región
   - Renderizar palabra completa centrada sobre el blur
   - Los dígrafos (CH, GH, etc.) ocupan espacio proporcional a su bounding box original
```

**Algoritmo de overlay:**
```
Para cada detección/palabra (box, class_id, confidence):
  1. Extraer región del bounding box (individual o agrupado)
  2. Aplicar GaussianBlur (sigma fijo: 12px)
  3. Dibujar rectángulo semi-transparente (opcional, para legibilidad)
  4. Calcular tamaño de fuente proporcional al alto del box
  5. Renderizar letra traducida centrada con color según confianza
  6. Renderizar glifo Keposhka original debajo en tamaño pequeño
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

**Mapeo de etiquetas a caracteres de fuente Keposhka:**

La fuente Keposhka usa caracteres especiales para representar los dígrafos:

```dart
// keposhka_font_map.dart
/// Mapea etiquetas de clase al carácter correspondiente en la fuente Keposhka.
/// Para letras y números es mapeo directo (A→A, a→a, 0→0).
/// Para dígrafos se usan caracteres especiales.
const keposhkaFontMap = {
  // Números - mapeo directo
  '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
  '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',

  // Letras - mapeo directo
  'A': 'A', 'a': 'a', 'B': 'B', 'b': 'b', 'C': 'C', 'c': 'c',
  'D': 'D', 'd': 'd', 'E': 'E', 'e': 'e', 'F': 'F', 'f': 'f',
  'G': 'G', 'g': 'g', 'H': 'H', 'h': 'h', 'I': 'I', 'i': 'i',
  'J': 'J', 'j': 'j', 'K': 'K', 'k': 'k', 'L': 'L', 'l': 'l',
  'M': 'M', 'm': 'm', 'N': 'N', 'n': 'n', 'O': 'O', 'o': 'o',
  'P': 'P', 'p': 'p', 'Q': 'Q', 'q': 'q', 'R': 'R', 'r': 'r',
  'S': 'S', 's': 's', 'T': 'T', 't': 't', 'U': 'U', 'u': 'u',
  'V': 'V', 'v': 'v', 'W': 'W', 'w': 'w', 'X': 'X', 'x': 'x',
  'Y': 'Y', 'y': 'y', 'Z': 'Z', 'z': 'z',

  // Dígrafos - caracteres especiales en la fuente
  'CH': '(',   // Paréntesis abierto
  'ch': ')',   // Paréntesis cerrado
  'GH': '\\',  // Backslash
  'gh': '|',   // Pipe
  'TH': '[',   // Corchete abierto
  'th': ']',   // Corchete cerrado
  'SH': '{',   // Llave abierta
  'sh': '}',   // Llave cerrada
};

/// Obtiene el carácter de fuente Keposhka para una etiqueta de clase
String getKeposhkaChar(String label) => keposhkaFontMap[label] ?? label;
```

**Uso en el ReferenceGrid:**
```dart
// Para mostrar un glifo Keposhka:
Text(
  getKeposhkaChar('CH'),  // Retorna '('
  style: TextStyle(fontFamily: 'Keposhka', fontSize: 32),
)
// La fuente renderiza '(' como el glifo Keposhka para "CH"
```

### 4. Pantalla de Verificación (VerificationScreen)

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
│  REFERENCIA (carrusel ~80px)           │ ← Carrusel inferior
│  [A|𐌀] [B|𐌁] [C|𐌂] [D|𐌃] [E|𐌄] ...  │   scroll horizontal
└────────────────────────────────────────┘
```

**Elementos:**
- Dos imágenes lado a lado (scroll sincronizado, zoom sincronizado máximo 4x)
- Carrusel de referencia inferior (~80px, compacto, scroll horizontal)
- El carrusel sirve para seleccionar cuando se está editando un carácter

**Interacciones:**
- Pinch to zoom (sincronizado en ambas imágenes, máximo 4x para ocultar artefactos del blur)
- Pan (sincronizado)

### 5. Carrusel de Referencia (ReferenceCarousel)

**Propósito:**
1. Referencia visual rápida de todos los caracteres
2. Selector para edición manual de detecciones incorrectas

**Layout:**
- Carrusel horizontal ~80px de altura
- Toggle button para mostrar/ocultar (no superpone, constriñe el espacio disponible)
- Cada celda muestra: letra latina arriba + glifo Keposhka abajo
- Scroll horizontal por todas las 70 clases

**Organización en carrusel:**
```
[A-Z + dígrafos mayúsculas] [a-z + dígrafos minúsculas] [0-9]
```
Los dígrafos se insertan en su posición alfabética: CH después de C, GH después de G, SH después de S, TH después de T.

**Interactividad:**
- En modo normal: solo scroll visual de referencia
- En modo edición (cuando hay un carácter seleccionado): tap para asignar esa clase al carácter seleccionado

### 6. Flujo de Edición Manual

**Propósito:** Permitir al usuario corregir detecciones incorrectas

**Flujo:**
1. Usuario toca un carácter detectado en la imagen
2. El carácter se resalta con borde/highlight de color
3. El carrusel de referencia se activa para selección
4. Usuario toca un carácter en el carrusel
5. Se muestra preview del cambio
6. Usuario confirma o cancela el cambio

**Indicador visual de selección:**
- Borde de color brillante alrededor del carácter seleccionado
- El resto de la imagen mantiene su estado normal

**Confirmación:**
- Al seleccionar del carrusel, se muestra preview del cambio
- Requiere segundo tap para confirmar
- Botón de cancelar para descartar

**Implementación:**
```dart
// reference_carousel.dart
class ReferenceCarousel extends StatelessWidget {
  final String? selectedLabel;  // Carácter actualmente en edición
  final ValueChanged<String>? onSelect;  // Callback al seleccionar para edición
  final bool isEditMode;  // Si hay un carácter seleccionado para editar

  // Orden de caracteres en el carrusel (3 secciones: mayúsculas, minúsculas, números)
  // Los dígrafos están integrados en su posición alfabética
  static const _allLabels = [
    // Mayúsculas (con dígrafos en posición alfabética)
    'A', 'B', 'C', 'CH', 'D', 'E', 'F', 'G', 'GH', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'SH', 'T', 'TH', 'U', 'V', 'W', 'X', 'Y', 'Z',
    // Minúsculas (con dígrafos en posición alfabética)
    'a', 'b', 'c', 'ch', 'd', 'e', 'f', 'g', 'gh', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 'sh', 't', 'th', 'u', 'v', 'w', 'x', 'y', 'z',
    // Números
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
  ];

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 80,
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        itemCount: _allLabels.length,
        itemBuilder: (context, index) {
          final label = _allLabels[index];
          return _buildCharCell(label);
        },
      ),
    );
  }

  Widget _buildCharCell(String label) {
    final isSelected = label == selectedLabel;
    return GestureDetector(
      onTap: isEditMode ? () => onSelect?.call(label) : null,
      child: Container(
        width: 50,
        margin: const EdgeInsets.symmetric(horizontal: 2),
        decoration: BoxDecoration(
          color: isSelected
              ? AppColors.keposhkaRed
              : AppColors.keposhkaSteel.withOpacity(0.3),
          border: Border.all(
            color: isEditMode ? AppColors.keposhkaRed : AppColors.keposhkaSteel,
            width: isEditMode ? 2 : 1,
          ),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Etiqueta latina
            Text(label, style: TextStyle(
              fontFamily: 'Oswald',
              fontSize: 12,
              color: isSelected ? AppColors.keposhkaWhite : AppColors.keposhkaCream,
            )),
            // Glifo Keposhka
            Text(
              getKeposhkaChar(label),
              style: TextStyle(
                fontFamily: 'Keposhka',
                fontSize: 24,
                color: isSelected ? AppColors.keposhkaWhite : AppColors.keposhkaCream,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
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

  # Almacenamiento (solo para caché de historial temporal)
  shared_preferences: ^2.2.2

  # Guardado a galería
  gal: ^2.3.0

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
    # Display - Constructivista (títulos grandes)
    - family: BebasNeue
      fonts:
        - asset: assets/fonts/BebasNeue-Regular.ttf

    # Headings - Industrial (subtítulos, UI)
    - family: Oswald
      fonts:
        - asset: assets/fonts/Oswald-Regular.ttf
        - asset: assets/fonts/Oswald-Medium.ttf
          weight: 500
        - asset: assets/fonts/Oswald-Bold.ttf
          weight: 700

    # Monospace - Datos, IDs, código
    - family: JetBrainsMono
      fonts:
        - asset: assets/fonts/JetBrainsMono-Regular.ttf

    # Keposhka - Glifos del idioma ficticio
    - family: Keposhka
      fonts:
        - asset: assets/fonts/KeposhkaRegular.otf

  assets:
    - assets/models/keposhka_detector.tflite
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

### Prioridades de rendimiento

- **Calidad sobre velocidad**: En dispositivos de gama baja, mantener calidad del blur aunque tarde más
- **Blur sigma fijo**: 12px para todos los caracteres, sin ajuste dinámico
- **Zoom máximo 4x**: Limitar zoom para ocultar artefactos del blur

## Conectividad

La aplicación funciona **100% offline**:
- Todas las fuentes embebidas en el APK
- Modelo TFLite incluido en assets
- Sin analytics ni funcionalidad que requiera red
- Sin actualizaciones remotas del modelo

## Fases de Desarrollo

### Fase 1: Setup y UI Base
- [ ] Crear proyecto Flutter
- [ ] Configurar tema brutalista (colores, tipografía)
- [ ] Implementar componentes base (BrutalButton, RegimePanel, etc.)
- [ ] Crear navegación con GoRouter
- [ ] Implementar HomeScreen con sección de recientes
- [ ] Agregar fuentes (Oswald, Bebas Neue, JetBrains Mono, Keposhka)
- [ ] Implementar carrusel de referencia (ReferenceCarousel)

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
- [ ] Implementar ImageService para blur (sigma fijo 12px)
- [ ] Implementar algoritmo de agrupación de palabras
- [ ] Renderizar letras con color coding (verde/amarillo/rojo)
- [ ] Mostrar glifo Keposhka pequeño debajo de cada traducción
- [ ] Crear TranslationScreen con imagen procesada
- [ ] Ajustar tamaño de fuente según box
- [ ] Optimizar rendimiento (isolate, priorizar calidad)

### Fase 5: Edición Manual
- [ ] Implementar selección de carácter (tap con highlight)
- [ ] Activar carrusel para selección de clase
- [ ] Implementar preview + confirmación de cambio
- [ ] Integrar flujo de edición con TranslationScreen

### Fase 6: Modo Verificación
- [ ] Implementar VerificationScreen
- [ ] Vista lado a lado sincronizada (zoom max 4x)
- [ ] Integrar carrusel de referencia
- [ ] Scroll/zoom sincronizado

### Fase 7: Historial y Guardado
- [ ] Implementar caché de traducciones recientes (10-20)
- [ ] Mostrar thumbnails en HomeScreen
- [ ] Implementar guardado a galería (PNG)
- [ ] Manejo de errores y fallback a modo manual

### Fase 8: Pulido y Testing
- [ ] Pruebas en múltiples dispositivos
- [ ] Verificar tamaño APK < 50MB
- [ ] Manejar edge cases (sin detecciones, errores TFLite)
- [ ] Agregar animaciones y transiciones brutalistas
- [ ] Testing de UI y UX
- [ ] Accesibilidad mínima (labels para screen readers)

### Fase 9: Release
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
| Tamaño APK | < 50MB (límite estricto) |
| RAM en uso | < 200MB |

## Accesibilidad

Implementación mínima viable:
- Labels accesibles para screen readers en todos los elementos interactivos
- Respetar tamaño de fuente del sistema cuando sea posible
- Sin tema claro alternativo ni adaptaciones de alto contraste

## Audio y Haptics

- Sin efectos de sonido
- Solo vibración háptica opcional (feedback del sistema)

## Manejo de Errores y Edge Cases

### Sin detecciones
Cuando el modelo no detecta ningún carácter:
- Mostrar mensaje simple: "No se detectaron caracteres Keposhka"
- Botón para reintentar con la misma imagen

### Error de modelo TFLite
Cuando falla la carga o inferencia del modelo:
- Fallback a modo manual: "No se pudo detectar automáticamente. Usa la referencia para traducir manualmente."
- El usuario puede usar el carrusel de referencia para interpretar la imagen visualmente

### Texto rotado/vertical
- La app asume texto horizontal
- Texto rotado se detectará carácter por carácter pero sin orden de lectura correcto
- El usuario puede usar la referencia para interpretar manualmente

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
Nota: Priorizar calidad sobre velocidad en todos los casos.
```

---

## Sistema de Confianza Visual

### Color Coding por Confianza

Cada carácter detectado se muestra con un color según su nivel de confianza:

| Rango | Color | Hex | Descripción |
|-------|-------|-----|-------------|
| > 80% | Verde | `#4A7C23` (successLight) | Alta confianza, detección segura |
| 50-80% | Amarillo | `#8B6914` (warning) | Confianza media, verificar |
| < 50% | Rojo | `#89181A` (keposhkaRed) | Baja confianza, probablemente incorrecto |

**Nota:** Solo se muestran detecciones con confianza >= 25% (threshold del modelo).

### Implementación del Color Coding

```dart
Color getConfidenceColor(double confidence) {
  if (confidence > 0.80) return AppColors.successLight;
  if (confidence > 0.50) return AppColors.warning;
  return AppColors.keposhkaRed;
}
```

---

## Flujo de Edición Detallado

### Estado de Edición

```dart
class EditState {
  final int? selectedDetectionIndex;  // null = no hay selección
  final String? previewLabel;         // Label en preview (antes de confirmar)
  final bool isConfirmationPending;   // Esperando confirmación
}
```

### Flujo Paso a Paso

1. **Estado inicial**: Usuario ve la imagen con traducciones
2. **Tap en carácter**:
   - `selectedDetectionIndex` = índice del carácter tocado
   - Se muestra highlight de color en ese carácter
   - El carrusel inferior entra en modo selección
3. **Tap en carrusel**:
   - `previewLabel` = label seleccionado del carrusel
   - Se muestra preview del cambio sobre la imagen
   - Aparecen botones "Confirmar" y "Cancelar"
4. **Confirmar**:
   - Se actualiza la detección con el nuevo label
   - Se regenera el overlay de la imagen
   - Se limpia el estado de edición
5. **Cancelar**:
   - Se descarta el `previewLabel`
   - El carácter sigue seleccionado para elegir otra opción

### Widget de Edición

```dart
class EditableOverlay extends StatelessWidget {
  final List<Detection> detections;
  final int? selectedIndex;
  final String? previewLabel;
  final ValueChanged<int> onSelect;
  final ValueChanged<String> onPreview;
  final VoidCallback onConfirm;
  final VoidCallback onCancel;

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // Imagen con overlay
        TranslationOverlay(
          detections: detections,
          highlightIndex: selectedIndex,
          previewLabel: previewLabel,
        ),
        // Carrusel inferior (si hay selección)
        if (selectedIndex != null)
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: ReferenceCarousel(
              isEditMode: true,
              selectedLabel: previewLabel ?? detections[selectedIndex!].label,
              onSelect: onPreview,
            ),
          ),
        // Botones de confirmación (si hay preview)
        if (previewLabel != null)
          Positioned(
            bottom: 90,
            right: 16,
            child: Row(
              children: [
                BrutalButton(text: 'CANCELAR', onPressed: onCancel, isPrimary: false),
                SizedBox(width: 8),
                BrutalButton(text: 'CONFIRMAR', onPressed: onConfirm),
              ],
            ),
          ),
      ],
    );
  }
}
```

---

## Almacenamiento del Historial

### Estructura de Caché

```dart
class TranslationCache {
  final String imagePath;           // Path temporal de la imagen
  final String translatedImagePath; // Path de la imagen con overlay
  final List<Detection> detections; // Detecciones (para re-editar)
  final DateTime timestamp;         // Para ordenar y limpiar
}
```

### Política de Limpieza

- Máximo 20 traducciones en caché
- Al agregar la #21, se elimina la más antigua
- Se guardan en directorio temporal de la app
- Se borran al desinstalar la app

### Persistencia

- Usar SharedPreferences para guardar lista de paths
- Las imágenes se guardan como archivos en el directorio de caché de la app
- No se persisten las detecciones (se re-procesan si el usuario reabre)

---

## Referencias

### Proyecto Keposhka
- Modelo YOLO: `keposhka-detector-model/specs/keposhka_detector_model.md`
- Mapeo de fuente Keposhka: `keposhka_labeling/frontend/src/components/ReferencePanel.tsx`

### Diseño UI (archivos base para adaptar a Flutter)
- Configuración de tema: `keposhka_labeling/frontend/tailwind.config.js`
- Estilos CSS: `keposhka_labeling/frontend/src/index.css`
- Fuentes: `keposhka_labeling/frontend/package.json`

### Documentación externa
- TFLite Flutter: https://pub.dev/packages/tflite_flutter
- YOLO TFLite export: Ultralytics docs
- Google Fonts (Bebas Neue, Oswald): https://fonts.google.com
- JetBrains Mono: https://www.jetbrains.com/lp/mono/
