import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:webview_flutter/webview_flutter.dart';

void main() {
  runApp(const FallDetectionApp());
}

class FallDetectionApp extends StatelessWidget {
  const FallDetectionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: '넘어짐 감지 카메라',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue,
        ),
        useMaterial3: true,
      ),
      home: const CameraScreen(),
    );
  }
}

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() {
    return _CameraScreenState();
  }
}

class _CameraScreenState extends State<CameraScreen> {
  /*
   * 반드시 본인 Mac IP와 Flask 포트로 변경하세요.
   *
   * 잘못된 예:
   * http://127.0.0.1:5001
   *
   * 스마트폰에서는 127.0.0.1이 스마트폰 자신을 의미합니다.
   */
  static const String serverUrl =
      'http://172.28.114.175:5001';

  late final WebViewController webViewController;

  Timer? statusTimer;

  bool serverConnected = false;
  bool cameraConnected = false;
  bool fallDetected = false;
  bool alertShowing = false;
  bool pageLoading = true;

  String label = 'waiting';
  String? detectedAt;
  String? cameraError;

  double confidence = 0.0;
  double? frameAge;

  @override
  void initState() {
    super.initState();

    webViewController = WebViewController()
      ..setJavaScriptMode(
        JavaScriptMode.unrestricted,
      )
      ..setBackgroundColor(Colors.black)
      ..setNavigationDelegate(
        NavigationDelegate(
          onPageStarted: (_) {
            if (!mounted) {
              return;
            }

            setState(() {
              pageLoading = true;
            });
          },
          onPageFinished: (_) {
            if (!mounted) {
              return;
            }

            setState(() {
              pageLoading = false;
            });
          },
          onWebResourceError: (error) {
            debugPrint(
              'WebView 오류: ${error.description}',
            );

            if (!mounted) {
              return;
            }

            setState(() {
              pageLoading = false;
            });
          },
        ),
      )
      ..loadRequest(
        Uri.parse(serverUrl),
      );

    _startStatusPolling();
  }

  void _startStatusPolling() {
    _checkStatus();

    statusTimer = Timer.periodic(
      const Duration(seconds: 1),
      (_) {
        _checkStatus();
      },
    );
  }

  Future<void> _checkStatus() async {
    try {
      final response = await http
          .get(
            Uri.parse('$serverUrl/status'),
          )
          .timeout(
            const Duration(seconds: 3),
          );

      if (response.statusCode != 200) {
        throw Exception(
          '서버 응답 코드 ${response.statusCode}',
        );
      }

      final decoded = jsonDecode(response.body);

      if (decoded is! Map<String, dynamic>) {
        throw const FormatException(
          '잘못된 서버 응답 형식입니다.',
        );
      }

      if (!mounted) {
        return;
      }

      final newFallDetected =
          decoded['fall_detected'] == true;

      setState(() {
        serverConnected =
            decoded['server_connected'] == true;

        cameraConnected =
            decoded['camera_connected'] == true;

        fallDetected = newFallDetected;

        label =
            decoded['label']?.toString() ?? 'unknown';

        confidence =
            (decoded['confidence'] as num?)
                    ?.toDouble() ??
                0.0;

        frameAge =
            (decoded['frame_age'] as num?)
                ?.toDouble();

        detectedAt =
            decoded['detected_at']?.toString();

        cameraError =
            decoded['camera_error']?.toString();
      });

      if (newFallDetected && !alertShowing) {
        _showFallAlert();
      }
    } catch (error) {
      debugPrint('상태 확인 오류: $error');

      if (!mounted) {
        return;
      }

      setState(() {
        serverConnected = false;
        cameraConnected = false;
        cameraError = error.toString();
      });
    }
  }

  Future<void> _showFallAlert() async {
    if (!mounted || alertShowing) {
      return;
    }

    alertShowing = true;

    await showDialog<void>(
      context: context,
      barrierDismissible: false,
      builder: (dialogContext) {
        return AlertDialog(
          icon: const Icon(
            Icons.warning_amber_rounded,
            color: Colors.red,
            size: 60,
          ),
          title: const Text(
            '넘어짐이 감지되었습니다',
            textAlign: TextAlign.center,
          ),
          content: Text(
            '감지 시간: ${detectedAt ?? "-"}\n'
            '현재 상태: ${_labelText(label)}\n'
            '신뢰도: '
            '${(confidence * 100).toStringAsFixed(1)}%',
            textAlign: TextAlign.center,
          ),
          actionsAlignment:
              MainAxisAlignment.center,
          actions: [
            FilledButton(
              onPressed: () {
                Navigator.of(dialogContext).pop();
              },
              child: const Text('확인'),
            ),
          ],
        );
      },
    );

    await _resetFall();

    alertShowing = false;
  }

  Future<void> _resetFall() async {
    try {
      final response = await http
          .post(
            Uri.parse('$serverUrl/reset-fall'),
          )
          .timeout(
            const Duration(seconds: 3),
          );

      if (response.statusCode != 200) {
        throw Exception(
          '초기화 실패: ${response.statusCode}',
        );
      }

      if (!mounted) {
        return;
      }

      setState(() {
        fallDetected = false;
        detectedAt = null;
      });

      await _checkStatus();
    } catch (error) {
      if (!mounted) {
        return;
      }

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            '넘어짐 경고 초기화 실패: $error',
          ),
        ),
      );
    }
  }

  Future<void> _reconnect() async {
    setState(() {
      pageLoading = true;
    });

    await webViewController.loadRequest(
      Uri.parse(
        '$serverUrl/?time='
        '${DateTime.now().millisecondsSinceEpoch}',
      ),
    );

    await _checkStatus();
  }

  String _labelText(String value) {
    switch (value) {
      case 'normal':
        return '정상';
      case 'sitting':
        return '앉음';
      case 'lying':
        return '누움';
      case 'fall':
        return '넘어짐';
      case 'no_person':
        return '사람 없음';
      case 'waiting':
        return '감지 준비 중';
      default:
        return value;
    }
  }

  Color get statusColor {
    if (!serverConnected || !cameraConnected) {
      return Colors.red;
    }

    if (fallDetected) {
      return Colors.red;
    }

    return Colors.green;
  }

  String get connectionText {
    if (!serverConnected) {
      return '컴퓨터 서버 연결 안 됨';
    }

    if (!cameraConnected) {
      return '카메라 연결 끊김';
    }

    if (fallDetected) {
      return '넘어짐 감지';
    }

    return '영상 수신 중';
  }

  String get detailText {
    if (!serverConnected) {
      return 'Mac IP, 포트, 와이파이를 확인하세요.';
    }

    if (!cameraConnected) {
      return cameraError ?? '카메라 상태를 확인하세요.';
    }

    return '현재 상태: ${_labelText(label)}'
        ' · 신뢰도 '
        '${(confidence * 100).toStringAsFixed(1)}%'
        '\n마지막 프레임: '
        '${frameAge?.toStringAsFixed(2) ?? "-"}초 전';
  }

  @override
  void dispose() {
    statusTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          '넘어짐 감지 카메라',
        ),
        centerTitle: true,
        actions: [
          IconButton(
            onPressed: _reconnect,
            icon: const Icon(Icons.refresh),
            tooltip: '다시 연결',
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              flex: 3,
              child: Stack(
                children: [
                  Positioned.fill(
                    child: ColoredBox(
                      color: Colors.black,
                      child: WebViewWidget(
                        controller:
                            webViewController,
                      ),
                    ),
                  ),
                  if (pageLoading)
                    const Positioned.fill(
                      child: ColoredBox(
                        color: Colors.black54,
                        child: Center(
                          child:
                              CircularProgressIndicator(),
                        ),
                      ),
                    ),
                ],
              ),
            ),
            Expanded(
              flex: 2,
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    Card(
                      child: ListTile(
                        leading: Icon(
                          cameraConnected
                              ? Icons.videocam
                              : Icons.videocam_off,
                          color: statusColor,
                          size: 34,
                        ),
                        title: Text(
                          connectionText,
                          style: TextStyle(
                            color: statusColor,
                            fontWeight:
                                FontWeight.bold,
                          ),
                        ),
                        subtitle: Text(detailText),
                      ),
                    ),
                    const SizedBox(height: 12),
                    SizedBox(
                      width: double.infinity,
                      child: FilledButton.icon(
                        onPressed: _reconnect,
                        icon:
                            const Icon(Icons.refresh),
                        label:
                            const Text('다시 연결'),
                      ),
                    ),
                    if (fallDetected) ...[
                      const SizedBox(height: 12),
                      SizedBox(
                        width: double.infinity,
                        child: FilledButton.icon(
                          style:
                              FilledButton.styleFrom(
                            backgroundColor:
                                Colors.red,
                          ),
                          onPressed: _resetFall,
                          icon: const Icon(
                            Icons.warning,
                          ),
                          label: const Text(
                            '넘어짐 경고 확인',
                          ),
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}