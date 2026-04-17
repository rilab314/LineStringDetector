---
name: config 상수 직접 참조
description: cfg 모듈에서 이미 제공하는 상수를 중간 변수로 재정의하지 않는다
type: feedback
---

`cfg.DATA_ROOT`, `cfg.RESULT_PATH` 등 config에서 이미 정의된 상수를 모듈 레벨에서 `DATA_ROOT = cfg.DATA_ROOT` 형태로 재선언하지 말 것.

**Why:** 불필요한 중간 상수는 코드 중복이고 가독성을 해친다.

**How to apply:** config 값은 항상 `cfg.XXX`로 직접 참조한다.
