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

package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"slices"
	"sort"
)

// Check reports every way patched differs from base beyond catalogue data.
// Allowed: changed literal values and comments, new constants with literal
// values, and added or removed entries in a function whose body is a single
// `return <composite literal>`. Everything else is reported.
func Check(base, patched []byte) []string {
	fset := token.NewFileSet()

	baseFile, err := parser.ParseFile(fset, "base.go", base, parser.SkipObjectResolution)
	if err != nil {
		return []string{"parse base: " + err.Error()}
	}

	patchedFile, err := parser.ParseFile(fset, "patched.go", patched, parser.SkipObjectResolution)
	if err != nil {
		return []string{"parse patched: " + err.Error()}
	}

	var problems []string
	problems = append(problems, compareImports(baseFile, patchedFile)...)
	problems = append(problems, compareFuncs(fset, baseFile, patchedFile)...)
	problems = append(problems, compareFuncLits(fset, baseFile, patchedFile)...)
	problems = append(problems, compareNames(fset, baseFile, patchedFile)...)
	problems = append(problems, checkConsts(patchedFile)...)
	problems = append(problems, compareCalls(baseFile, patchedFile)...)

	return problems
}

func render(fset *token.FileSet, node ast.Node) string {
	var buf bytes.Buffer
	if err := printer.Fprint(&buf, fset, node); err != nil {
		return "<unprintable: " + err.Error() + ">"
	}

	return buf.String()
}

func importSet(file *ast.File) []string {
	imports := make([]string, 0, len(file.Imports))

	for _, spec := range file.Imports {
		name := ""
		if spec.Name != nil {
			name = spec.Name.Name + " "
		}

		imports = append(imports, name+spec.Path.Value)
	}

	sort.Strings(imports)

	return imports
}

func compareImports(base, patched *ast.File) []string {
	before, after := importSet(base), importSet(patched)
	if slices.Equal(before, after) {
		return nil
	}

	return []string{fmt.Sprintf("imports changed: %v -> %v", before, after)}
}

func funcDecls(file *ast.File) map[string]*ast.FuncDecl {
	funcs := map[string]*ast.FuncDecl{}

	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok {
			funcs[fn.Name.Name] = fn
		}
	}

	return funcs
}

// isDataReturn reports whether fn's body is exactly `return <composite literal>`.
func isDataReturn(fn *ast.FuncDecl) bool {
	if fn.Body == nil || len(fn.Body.List) != 1 {
		return false
	}

	ret, ok := fn.Body.List[0].(*ast.ReturnStmt)
	if !ok || len(ret.Results) != 1 {
		return false
	}

	_, ok = ret.Results[0].(*ast.CompositeLit)

	return ok
}

func compareFuncs(fset *token.FileSet, base, patched *ast.File) []string {
	before, after := funcDecls(base), funcDecls(patched)
	var problems []string

	for name, fn := range after {
		old, ok := before[name]
		if !ok {
			problems = append(problems, "new function "+name)
			continue
		}

		if render(fset, old.Type) != render(fset, fn.Type) {
			problems = append(problems, "signature changed: "+name)
		}

		if isDataReturn(old) && isDataReturn(fn) {
			continue
		}

		if render(fset, old.Body) != render(fset, fn.Body) {
			problems = append(problems, "body changed: "+name)
		}
	}

	for name := range before {
		if _, ok := after[name]; !ok {
			problems = append(problems, "removed function "+name)
		}
	}

	sort.Strings(problems)

	return problems
}

func funcLits(fset *token.FileSet, file *ast.File) []string {
	var lits []string

	ast.Inspect(file, func(node ast.Node) bool {
		if lit, ok := node.(*ast.FuncLit); ok {
			lits = append(lits, render(fset, lit))
		}

		return true
	})
	sort.Strings(lits)

	return lits
}

func compareFuncLits(fset *token.FileSet, base, patched *ast.File) []string {
	if slices.Equal(funcLits(fset, base), funcLits(fset, patched)) {
		return nil
	}

	return []string{"function literals changed"}
}

// varNames lists the names of top-level vars.
func varNames(file *ast.File) []string {
	var names []string

	for _, decl := range file.Decls {
		gen, ok := decl.(*ast.GenDecl)
		if !ok || gen.Tok != token.VAR {
			continue
		}

		for _, spec := range gen.Specs {
			if value, ok := spec.(*ast.ValueSpec); ok {
				for _, ident := range value.Names {
					names = append(names, ident.Name)
				}
			}
		}
	}

	sort.Strings(names)

	return names
}

// typeSources renders every top-level type declaration.
func typeSources(fset *token.FileSet, file *ast.File) []string {
	var types []string

	for _, decl := range file.Decls {
		gen, ok := decl.(*ast.GenDecl)
		if !ok || gen.Tok != token.TYPE {
			continue
		}

		for _, spec := range gen.Specs {
			types = append(types, render(fset, spec))
		}
	}

	sort.Strings(types)

	return types
}

func compareNames(fset *token.FileSet, base, patched *ast.File) []string {
	var problems []string

	beforeVars, afterVars := varNames(base), varNames(patched)
	if !slices.Equal(beforeVars, afterVars) {
		problems = append(problems, fmt.Sprintf("vars changed: %v -> %v", beforeVars, afterVars))
	}

	if !slices.Equal(typeSources(fset, base), typeSources(fset, patched)) {
		problems = append(problems, "types changed")
	}

	return problems
}

func checkConsts(patched *ast.File) []string {
	var problems []string

	for _, decl := range patched.Decls {
		gen, ok := decl.(*ast.GenDecl)
		if !ok || gen.Tok != token.CONST {
			continue
		}

		for _, spec := range gen.Specs {
			value, ok := spec.(*ast.ValueSpec)
			if !ok {
				continue
			}

			for index, expr := range value.Values {
				switch expr.(type) {
				case *ast.BasicLit, *ast.Ident, *ast.SelectorExpr:
				default:
					name := value.Names[min(index, len(value.Names)-1)].Name
					problems = append(problems, "const "+name+" has a non-literal value")
				}
			}
		}
	}

	return problems
}

func callName(fun ast.Expr) string {
	switch typed := fun.(type) {
	case *ast.Ident:
		return typed.Name
	case *ast.SelectorExpr:
		if pkg, ok := typed.X.(*ast.Ident); ok {
			return pkg.Name + "." + typed.Sel.Name
		}

		return "." + typed.Sel.Name
	case *ast.IndexExpr:
		return callName(typed.X)
	default:
		return fmt.Sprintf("%T", fun)
	}
}

func callNames(file *ast.File) map[string]bool {
	names := map[string]bool{}

	ast.Inspect(file, func(node ast.Node) bool {
		if call, ok := node.(*ast.CallExpr); ok {
			names[callName(call.Fun)] = true
		}

		return true
	})

	return names
}

// compareCalls rejects any call the base file doesn't already make.
func compareCalls(base, patched *ast.File) []string {
	before := callNames(base)
	var problems []string

	for name := range callNames(patched) {
		if !before[name] {
			problems = append(problems, "new call: "+name)
		}
	}

	sort.Strings(problems)

	return problems
}
