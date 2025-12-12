import React from 'react';
import {DocProvider} from '@docusaurus/theme-common/internal';
import DocItem from '@theme/DocItem';

// Custom layout without navbar for documentation pages
export default function CustomDocPage(props) {
  const {content: DocContent, versionMetadata} = props;

  if (DocContent) {
    return (
      <DocProvider content={DocContent}>
        <div className="doc-wrapper">
          <DocItem content={DocContent} />
        </div>
      </DocProvider>
    );
  }

  return null;
}