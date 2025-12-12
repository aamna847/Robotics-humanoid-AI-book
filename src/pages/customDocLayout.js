import React from 'react';
import {DocProvider} from '@docusaurus/theme-common/internal';
import DocItem from '@theme/DocItem';

// Custom layout without navbar for documentation pages
export default function CustomDocPage(props) {
  const {route, versionMetadata, location, sidebarName} = props;
  
  return (
    <DocProvider
      route={route}
      versionMetadata={versionMetadata}
      location={location}
      sidebarName={sidebarName}>
      <div className="doc-wrapper">
        <DocItem />
      </div>
    </DocProvider>
  );
}