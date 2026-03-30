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
	"cmp"
	"reflect"
	"slices"
	"strings"
	"testing"
)

var testingTType = reflect.TypeFor[*testing.T]()

type discoveredMethod struct {
	name   string
	method reflect.Method
}

// Run discovers exported Test* methods on a suite and executes them as parallel subtests.
//
// A suite must be a pointer to a struct whose test methods have the signature:
//
//	func (s *Suite) TestXxx(t *testing.T)
func Run(t *testing.T, suite any) {
	t.Helper()

	rv := reflect.ValueOf(suite)

	methods := discoverMethods(rv)
	if len(methods) == 0 {
		t.Fatalf("testsuite %T has no Test*(*testing.T) methods", suite)
	}

	for _, method := range methods {
		t.Run(method.name, func(t *testing.T) {
			t.Parallel()
			method.method.Func.Call([]reflect.Value{rv, reflect.ValueOf(t)})
		})
	}
}

func discoverMethods(rv reflect.Value) []discoveredMethod {
	if !rv.IsValid() {
		panic("testsuite: nil suite") //nolint:forbidigo // programmer error, not runtime
	}

	rt := rv.Type()
	if rt.Kind() != reflect.Pointer || rt.Elem().Kind() != reflect.Struct {
		panic("testsuite: suite must be a pointer to a struct") //nolint:forbidigo // programmer error, not runtime
	}

	var methods []discoveredMethod

	for method := range rt.Methods() {
		if method.PkgPath != "" {
			continue
		}

		if !strings.HasPrefix(method.Name, "Test") {
			continue
		}

		if method.Type.NumIn() != 2 || method.Type.NumOut() != 0 || method.Type.In(1) != testingTType {
			continue
		}

		methods = append(methods, discoveredMethod{
			name:   method.Name,
			method: method,
		})
	}

	slices.SortFunc(methods, func(a, b discoveredMethod) int {
		return cmp.Compare(a.name, b.name)
	})

	return methods
}
