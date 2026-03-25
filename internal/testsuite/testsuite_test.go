// Copyright 2026 Redpanda Data, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package testsuite

import (
	"reflect"
	"sync/atomic"
	"testing"
)

type baseSuite struct {
	count *atomic.Int32
}

func (s *baseSuite) TestBase(t *testing.T) { //nolint:paralleltest // parallelism managed by testsuite.Run
	t.Helper()
	s.count.Add(1)
}

func (*baseSuite) helper(*testing.T) {} //nolint:unused // intentionally unused, verifies discovery exclusion

type outerSuite struct {
	*baseSuite
}

func (s *outerSuite) TestOuter(t *testing.T) { //nolint:paralleltest // parallelism managed by testsuite.Run
	t.Helper()
	s.count.Add(1)
}

func (*outerSuite) NotATest(*testing.T) {}

func TestDiscoverMethods(t *testing.T) {
	t.Parallel()

	methods := discoverMethods(reflect.ValueOf(&outerSuite{
		baseSuite: &baseSuite{},
	}))

	got := make([]string, 0, len(methods))
	for _, method := range methods {
		got = append(got, method.name)
	}

	want := []string{"TestBase", "TestOuter"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected methods: got %v want %v", got, want)
	}
}

func TestRunExecutesAllTests(t *testing.T) { //nolint:tparallel // subtests parallelized by testsuite.Run
	t.Parallel()

	var count atomic.Int32

	ok := t.Run("suite", func(t *testing.T) {
		Run(t, &outerSuite{
			baseSuite: &baseSuite{count: &count},
		})
	})
	if !ok {
		t.Fatal("suite run failed")
	}

	if got := count.Load(); got != 2 {
		t.Fatalf("unexpected test count: got %d want 2", got)
	}
}
