use rumoca_ir_ast as ast;

/// Check if a component has annotation(Evaluate=true).
///
/// MLS §18.3: The Evaluate annotation indicates that a parameter should be
/// evaluated at compile time. This is used for structural parameters that
/// affect equation structure (e.g., if-equation branch selection).
///
/// Returns true if:
/// - The component has `annotation(Evaluate=true)`, or
/// - The component is declared `final` (implies compile-time evaluation)
pub(crate) fn has_evaluate_annotation(comp: &ast::Component) -> bool {
    if comp.is_final {
        return true;
    }

    comp.annotation.iter().any(is_evaluate_true_annotation)
}

fn is_evaluate_true_annotation(anno_expr: &ast::Expression) -> bool {
    let (name_text, value) = match anno_expr {
        ast::Expression::NamedArgument { name, value, .. } => (name.text.as_ref(), value.as_ref()),
        ast::Expression::Modification { target, value, .. } => {
            let Some(first_part) = target.parts.first() else {
                return false;
            };
            (first_part.ident.text.as_ref(), value.as_ref())
        }
        _ => return false,
    };

    if name_text != "Evaluate" {
        return false;
    }
    matches!(
        value,
        ast::Expression::Terminal {
            terminal_type: ast::TerminalType::Bool,
            token,
            ..
        } if token.text.as_ref() == "true"
    )
}

/// Check if a component has annotation(__rumoca(trainable=true)).
///
/// The `__rumoca(...)` vendor annotation namespaces rumoca-specific flags
/// (mirroring the experiment-settings idiom). The `trainable` flag marks a
/// parameter as a learnable weight (`theta`) rather than a fixed parameter
/// (`p`) so later codegen can split them apart.
///
/// Returns true only when an inner `trainable = true` modification is present
/// inside a (case-insensitive) `__rumoca(...)` wrapper.
pub(crate) fn has_trainable_annotation(comp: &ast::Component) -> bool {
    comp.annotation.iter().any(is_rumoca_trainable_annotation)
}

/// Returns the last identifier of a component reference, if any.
fn component_ref_last_ident(comp_ref: &ast::ComponentReference) -> Option<&str> {
    comp_ref
        .parts
        .last()
        .map(|part| part.ident.text.as_ref())
}

/// Unwrap a `__rumoca(...)` wrapper (case-insensitive) and look for an inner
/// `trainable = true`. Handles the four annotation Expression shapes the
/// experiment-settings extractor handles.
fn is_rumoca_trainable_annotation(anno_expr: &ast::Expression) -> bool {
    match anno_expr {
        ast::Expression::ClassModification {
            target,
            modifications,
            ..
        } if component_ref_last_ident(target)
            .is_some_and(|key| key.eq_ignore_ascii_case("__rumoca")) =>
        {
            modifications.iter().any(is_trainable_true_modification)
        }
        ast::Expression::FunctionCall { comp, args, .. }
            if component_ref_last_ident(comp)
                .is_some_and(|key| key.eq_ignore_ascii_case("__rumoca")) =>
        {
            args.iter().any(is_trainable_true_modification)
        }
        ast::Expression::NamedArgument { name, value, .. }
            if name.text.as_ref().eq_ignore_ascii_case("__rumoca") =>
        {
            is_rumoca_trainable_annotation(value)
        }
        ast::Expression::Modification { target, value, .. }
            if target
                .parts
                .last()
                .is_some_and(|part| part.ident.text.as_ref().eq_ignore_ascii_case("__rumoca")) =>
        {
            is_rumoca_trainable_annotation(value)
        }
        _ => false,
    }
}

/// Match an inner `trainable = true` modification, mirroring the
/// `is_evaluate_true_annotation` terminal-bool match.
fn is_trainable_true_modification(modification: &ast::Expression) -> bool {
    let (name_text, value) = match modification {
        ast::Expression::NamedArgument { name, value, .. } => (name.text.as_ref(), value.as_ref()),
        ast::Expression::Modification { target, value, .. } => {
            let Some(first_part) = target.parts.first() else {
                return false;
            };
            (first_part.ident.text.as_ref(), value.as_ref())
        }
        _ => return false,
    };

    if !name_text.eq_ignore_ascii_case("trainable") {
        return false;
    }
    matches!(
        value,
        ast::Expression::Terminal {
            terminal_type: ast::TerminalType::Bool,
            token,
            ..
        } if token.text.as_ref() == "true"
    )
}
